"""Known-answer metrics and end-to-end training/checkpoint safety tests.

Only temporary synthetic NIfTI fixtures are used, never real held-out data.
"""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np
import torch

from dataset import build_inventory, make_patient_split
from experiment import hard_overlap, aggregate_cases
from train import main as train_main, training_slice_order
from predict import main as predict_main


class MetricTests(unittest.TestCase):
    def test_known_overlap_and_empty_rules(self):
        truth = torch.tensor([0, 5, 5, 0])
        predicted = torch.tensor([0, 5, 0, 5])
        result = hard_overlap(predicted, truth)
        self.assertEqual(result['dice'][5], .5)
        self.assertAlmostEqual(result['iou'][5], 1/3)
        self.assertIsNone(result['dice'][1])
        self.assertEqual(hard_overlap(torch.tensor([5]), torch.tensor([0]))['dice'][5], 0)
        self.assertEqual(hard_overlap(truth, truth)['dice'][5], 1)

    def test_patient_mean_differs_from_scan_mean(self):
        cases = [dict(patient_id=p, dice=[score]*6, iou=[score]*6)
                 for p, score in [('A', 1.), ('A', 1.), ('A', 1.), ('B', 0.)]]
        report = aggregate_cases(cases)
        self.assertEqual(report['case_macro_dice'][5], .75)
        self.assertEqual(report['patient_macro_dice'][5], .5)


class SliceSamplingTests(unittest.TestCase):
    def test_balanced_batches_retain_negatives_and_step_budget(self):
        mask = torch.zeros(64, 4, 4, dtype=torch.long)
        mask[9, 1, 1] = 5
        a = training_slice_order(mask, torch.Generator().manual_seed(42), 'prostate-balanced')
        b = training_slice_order(mask, torch.Generator().manual_seed(42), 'prostate-balanced')
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(len(a), 64)
        flags = (mask[a] == 5).flatten(1).any(1)
        self.assertTrue(torch.equal(flags.reshape(-1, 8).sum(1), torch.full((8,), 4)))
        # The sole positive necessarily repeats; negative slices remain present.
        self.assertEqual(int((a == 9).sum()), 32)
        self.assertEqual(len(a[1::2].unique()), 32)

    def test_all_and_single_group_fallback_visit_every_slice(self):
        for fill, strategy in [(0, 'all'), (0, 'prostate-balanced'), (5, 'prostate-balanced')]:
            mask = torch.full((7, 2, 2), fill, dtype=torch.long)
            order = training_slice_order(mask, torch.Generator().manual_seed(3), strategy)
            self.assertTrue(torch.equal(order.sort().values, torch.arange(7)))
        with self.assertRaises(ValueError):
            training_slice_order(torch.zeros(0, 2, 2), torch.Generator())


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for folder in ['semantic_MRs', 'semantic_labels_only']:
            (self.root/folder).mkdir()
        x, y, z = np.indices((16, 16, 4)).astype(np.float32)
        self.pixels = x + .1*y + .2*z
        self.labels = np.broadcast_to((np.arange(16)*6//16)[:, None, None], (16, 16, 4)).copy().astype(np.int16)
        for patient in ['A001', 'B002', 'C003']:
            self.write_case(patient+'_Week0')
        self.manifest = make_patient_split(build_inventory(self.root), val_patients=1, test_patients=1)
        self.path = self.root/'split.json'
        self.path.write_text(json.dumps(self.manifest))
        self.common = ['--device', 'cpu', '--manifest', str(self.path), '--data-root', str(self.root)]
        self.training = ['--shape', '4', '16', '16', '--base-channels', '2', '--batch-size', '2']

    def write_case(self, case):
        nib.save(nib.Nifti1Image(self.pixels, np.eye(4)), self.root/'semantic_MRs'/f'{case}_LFOV.nii.gz')
        nib.save(nib.Nifti1Image(self.labels, np.eye(4)), self.root/'semantic_labels_only'/f'{case}_SEMANTIC.nii.gz')

    def remove_split_files(self, name):
        for scan in self.manifest['splits'][name]['scans']:
            Path(scan['image_path']).unlink()
            Path(scan['mask_path']).unlink()

    def quiet(self, fn, argv):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(argv)

    def test_debug_fits_and_reloads_without_heldout_files(self):
        self.remove_split_files('test')
        self.remove_split_files('validation')
        out = self.root/'debug'
        code = self.quiet(train_main, self.common+self.training+[
            '--mode', 'debug', '--debug-steps', '40', '--output', str(out)])
        self.assertEqual(code, 0)
        report = json.loads((out/'debug_report.json').read_text())
        self.assertTrue(report['passed'])
        self.assertLess(report['final_loss'], report['initial_loss'])
        self.assertTrue((out/'curves.png').is_file())
        pred = self.root/'preview'
        self.assertEqual(self.quiet(predict_main, self.common+['--split', 'train', '--checkpoint', str(out/'debug.pt'), '--output', str(pred)]), 0)
        files = list(pred.glob('*_prediction_RAS.nii.gz'))
        self.assertEqual(len(files), 1)
        saved = nib.load(files[0])
        self.assertEqual(saved.shape, (16, 16, 4))
        self.assertEqual(nib.aff2axcodes(saved.affine), ('R', 'A', 'S'))
        with self.assertRaises(ValueError):
            self.quiet(predict_main, self.common+['--split', 'validation', '--checkpoint', str(out/'debug.pt'), '--output', str(self.root/'blocked')])
        with self.assertRaises(ValueError):
            self.quiet(train_main, self.common+['--mode', 'test', '--final-test', '--checkpoint', str(out/'debug.pt'), '--output', str(self.root/'blocked-test')])
        with self.assertRaises(FileExistsError):
            self.quiet(train_main, self.common+self.training+['--output', str(out)])

    def test_fit_resume_and_explicit_test_gate(self):
        self.remove_split_files('test')
        first, resumed = self.root/'fit', self.root/'resumed'
        args = self.common+self.training+['--mode', 'fit']
        self.assertEqual(self.quiet(train_main, args+['--epochs', '1', '--output', str(first)]), 0)
        # The retained best can legitimately come from a different host/job.
        best = torch.load(first/'best.pt', weights_only=True)
        best['config']['hostname'] = 'earlier-host'
        torch.save(best, first/'best.pt')
        self.assertEqual(self.quiet(train_main, args+['--epochs', '2', '--resume', str(first/'last.pt'), '--output', str(resumed)]), 0)
        self.assertTrue((resumed/'best.pt').is_file())
        self.assertEqual(len(json.loads((resumed/'history.json').read_text())), 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            train_main(self.common+['--mode', 'test', '--checkpoint', str(resumed/'best.pt'), '--output', str(self.root/'blocked')])
        for scan in self.manifest['splits']['test']['scans']:
            self.write_case(scan['case_id'])
        result = self.root/'test'
        self.assertEqual(self.quiet(train_main, self.common+['--mode', 'test', '--final-test', '--checkpoint', str(resumed/'best.pt'), '--output', str(result)]), 0)
        report = json.loads((result/'test_metrics.json').read_text())
        self.assertEqual(report['case_count'], 1)
        self.assertEqual(report['cases'][0]['case_id'], self.manifest['splits']['test']['scans'][0]['case_id'])
        config = json.loads((result/'config.json').read_text())
        self.assertEqual(config['shape_dhw'], [4, 16, 16])
        self.assertEqual(config['base_channels'], 2)

    def test_balanced_fit_records_sampling_and_refuses_changed_resume(self):
        self.remove_split_files('test')
        # Only one of four slices has prostate: balanced sampling must repeat it.
        labels = np.zeros((16, 16, 4), dtype=np.int16)
        labels[4:12, 4:12, 0] = 5
        for name in ['train', 'validation']:
            for scan in self.manifest['splits'][name]['scans']:
                nib.save(nib.Nifti1Image(labels, np.eye(4)),
                         self.root/'semantic_labels_only'/f'{scan["case_id"]}_SEMANTIC.nii.gz')
        out = self.root/'balanced'
        args = self.common+self.training+['--mode', 'fit', '--slice-sampling', 'prostate-balanced']
        self.assertEqual(self.quiet(train_main, args+['--epochs', '1', '--output', str(out)]), 0)
        history = json.loads((out/'history.json').read_text())
        self.assertEqual(history[0]['sampled_slices'], 4)
        self.assertEqual(history[0]['sampled_prostate_slices'], 2)
        # A continuation cannot silently change sampling and claim the same run.
        with self.assertRaisesRegex(ValueError, 'sampling'):
            self.quiet(train_main, self.common+self.training+[
                '--mode', 'fit', '--resume', str(out/'last.pt'), '--epochs', '2',
                '--output', str(self.root/'wrong-resume')])
        self.assertEqual(self.quiet(train_main, args+['--resume', str(out/'last.pt'),
            '--epochs', '2', '--output', str(self.root/'balanced-resume')]), 0)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
