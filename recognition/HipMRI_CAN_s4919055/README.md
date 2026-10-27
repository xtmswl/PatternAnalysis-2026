# HipMRI Prostate Segmentation: CAN3D versus a 2D U-Net Baseline

Student: Xiaotian Mao

Student Number: s4919055

Course: COMP3710, Semester 2, 2026.

## Scope and actual status

On 15 October 2026 the student changed the selected route from Normal 2D CAN
to **Hard CAN3D**, retaining HipMRI anatomical segmentation. The main model
will segment a 3D MRI volume; a standard 2D U-Net will predict slices from the
same volumes as the baseline. Both will be evaluated on the same held-out
patients and spatial grid. This choice is permitted by Section 2.2, which
requires a baseline comparison; it does not require implementing every model
in the difficulty table. Hard permits up to 20/20 implementation marks and
40/40 overall, conditional on the work meeting all marking criteria/hurdles.

Current evidence: personal fork and topic-recognition branch exist. The local
brain PNG archive is excluded. Student terminal screenshots confirm access
to the course 3D directories, CPU allocation 662151 on vcpu-2, successful
PyTorch 2.14.0+cu130/nibabel 5.4.2 imports and an initial audit of B006_Week0.
The saved dataset.py was uploaded and run on the real B006_Week0 pair on
vcpu-2. The student screenshot confirms all four checks passed: both_3d,
same_shape, same_affine and finite_nonnegative_integer_labels. Local
synthetic checks also cover shape/affine mismatches and fractional labels.

The 2D U-Net baseline, soft Dice loss and baseline training/prediction drivers
have local synthetic checks. On 27 October, the student supplied the output
of real GPU debug job 700021: BASELINE_DEBUG_PASSED and
PREDICTION_PREVIEW_SAVED, with an empty displayed .err log. This establishes
short training and saved-weight inference on a training case, not generalization.
The first baseline fit, job 703884, completed 25 of 30 requested epochs before
the two-hour allocation expired. Its last validation report predicted no
bladder, rectum or prostate voxels; this is a failed organ-segmentation pilot,
not a successful baseline. CAN3D remains unimplemented; a usable trained
baseline and held-out testing remain pending.
On 26 October the student supplied the output of Slurm job 696947 on a100-8:
CUDA available=True, NVIDIA A100-PCIE-40GB, CUDA_CHECK_PASSED. This establishes
the tiny diagnostic's GPU execution, not segmentation accuracy or CAN3D memory.
The real filename inventory reports 211 MRI files, 211 mask files,
211 paired scans, 38 candidate patient IDs and zero naming/pairing issues.
The publisher label mapping is now verified. Filename-derived patient
identity interpretation remains an explicit filename-based assumption.
The full content audit completed remotely:211checked,211passed,0failed;
its downloaded report has been independently inspected and matches the
saved split case list and identity hash. The student plans to complete the
compulsory check-off on 27 October; completion has not yet been recorded.

## Research question

Does CAN3D's volumetric multi-scale context reduce prostate boundary and
slice-transition errors compared with a 2D U-Net, and what are the memory and
inference-time trade-offs? Neither model is assumed to win. Analyse both
held-out overlap scores and actual failure cases.

## Course and literature basis

The latest Ed attachment was checked on 26 October:
[COMP3710_Report_2026_v2.5.pdf](https://edstem.org/au/courses/38242/discussion/3600790).
Its cover says Version 2.5 Final Assessment Specification and still displays
DRAFT in the title. Sections 2.2 and 3-6 retain the relevant HipMRI requirements.
The model table on p. 6 lists CAN3D as Hard, reference [5], with downsampled
HipMRI 3D data and a test Dice target of at least 0.7. The U-Net baseline is
reference [1]. Appendix A, p. 17, specifies the data location. Sections 3-6
specify consultation, files, marking, Git/PR and AI disclosure.

Contact 7, pp. 16-17 and 22-28, teaches U-Net, dilation, CAN3D, mask encoding
and multiclass Dice. Contact 6 teaches mini-batch gradient descent,
backpropagation, Adam, preprocessing, validation and overfitting. The course
CNN notebook demonstrates an explicit PyTorch training loop and eval/no_grad.
These remain the training foundation after the difficulty change.

The main reference is Dai et al., CAN3D: Fast 3D Medical Image Segmentation
via Compact Context Aggregation. Its abstract and Sections 3.1 architecture
and training settings have been consulted; a full implementation-level reading
and architecture diagram check are still required before coding. Do not
replace it with the former Yu/Koltun 2D context module and call it CAN3D.
Document the selected 3D context blocks, normalisation, dilation, shortcuts,
downsampling and output-resolution recovery against the paper/lecture.

## Data pipeline and leakage prevention

- Course data root: /home/groups/comp3710/HipMRI_Study_open.
- Appendix A names semantic_MRs for 3D MRIs and semantic_labels_only for
  masks; both directories are confirmed by the student screenshot. Individual
  sample filenames are now confirmed. Match files by patient/week prefix,
  not directory-listing order. The full filename inventory and original-volume
  content audits passed. A three-training-case downsampling loader pilot also passed.
- Original collection: [CSIRO male pelvis MRI](https://doi.org/10.25919/45t8-p065).
  The course sheet describes 38 patients and 211 3D scans. The real Rangpur
  filename inventory now confirms 211 paired scans and 38 candidate IDs. Repeated scans from one patient must
  stay in the same split; volume/case ID is not automatically patient ID.
- Example from observed filenames: B006 Week0/Week1 scans belong to the same
  provisional patient group. Verify this against the full manifest or data
  description before finalising a split.
- Audit NIfTI shape, affine, orientation, voxel spacing, image-mask alignment,
  label values and anatomical mapping. Preserve these during preprocessing.
- Use aligned image interpolation and nearest-neighbour mask resampling for
  any justified downsampling. The target grid will follow data/memory evidence.
- PyTorch 3D images: B x 1 x D x H x W; logits/one-hot Dice targets:
  B x K x D x H x W. Verify the NIfTI-to-tensor axis conversion explicitly.
- Split by patient once and save the manifest. Derive baseline slices from
  the same preprocessed 3D volumes, preserving the exact patient split/grid.
  Reassemble baseline predictions for a consistent volumetric evaluation.
- Normalize images consistently using training-only fitted statistics where
  needed. Do not scale masks as continuous image intensities.

## Initial paired-data audit

Observed on 15 October from the student's initial audit and subsequent
execution of the saved dataset.py on Rangpur:

| Property | B006_Week0 MRI | B006_Week0 mask |
| --- | --- | --- |
| File suffix | LFOV.nii.gz | SEMANTIC.nii.gz |
| Shape | 256 x 256 x 128 | 256 x 256 x 128 |
| Voxel spacing | approximately 1.5625, 1.5625, 1.56 | same |
| Orientation | L, P, S | L, P, S |
| Stored dtype | int32 | int32 |
| Mask values | not applicable | 0, 1, 2, 3, 4, 5 |

Shape/affine comparisons returned True and label values were finite integers.
This establishes initial grid compatibility for one pair; it does not prove
full-dataset integrity or subtle anatomical boundary correctness. The
publisher documentation identifies prostate as code 5 (see mapping below).

The audit is a saved function in dataset.py. From the project folder on a
Rangpur compute node, run:

```bash
python dataset.py --case B006_Week0
```

To save a numeric-label visual preview on a compute node, run:

```bash
python dataset.py --case B006_Week0 --overlay runs/preview_B006.png
```

Matplotlib 3.11.2 was installed in the remote torch environment. The original
numeric-code preview was generated on Rangpur, downloaded and visually
inspected. Selected slices show no obvious gross image/mask displacement;
this does not validate all slices, subtle boundaries or the full dataset.
Both volumes are reoriented together to RAS for axial display. Each nonzero
code is shown at its largest-area axial slice; background code 0 is omitted.
Percentile contrast is display-only. Generated previews stay in ignored runs/.
The updated code uses publisher label names; a new named preview has not
been run remotely yet.

### Verified anatomical label mapping

The publisher's [BriefDataDescription.txt, collection v2](https://data.csiro.au/dap/ws/v2/collections/52549/support/5034),
retrieved 15 October 2026, specifies:

| Code | Class |
| --- | --- |
| 0 | Background |
| 1 | Body |
| 2 | Bones |
| 3 | Bladder |
| 4 | Rectum |
| 5 | Prostate |

The same document describes Week0 as pretreatment and Weeks1–7 as weekly
scans during radiotherapy. The updated audit also rejects codes outside 0–5.
The source mapping and original preview are consistent; full-dataset label
validation subsequently completed as part of the full211pair audit.
A shared-directory documentation search found
only the use agreement; the label description was retrieved from CSIRO's
public supporting-file API rather than inferred from anatomy or label order.

To use another dataset location, pass --data-root /path/to/HipMRI_Study_open.
The shared course path is on Rangpur, so the default command does not find
data on the Mac. Code is stored in the local Git repository and a working
copy must be transferred to Rangpur before remote execution. Terminal
commands launch saved code; the submitted implementation belongs in files.

## Full filename inventory

On a Rangpur compute node, run:

```bash
python dataset.py --inventory runs/inventory.json
```

This reports NIfTI image/mask counts, one-to-one matched scans, candidate
patients and issues. Pair by exact case stem, not list position. Both .nii
and .nii.gz files are indexed; duplicate cases, missing partners, unexpected
suffixes or patient/week patterns cause a nonzero exit status while saving
the report for inspection. File contents are not loaded or validated in this
mode. B006_Week0 and B006_Week1 are grouped under candidate ID B006; filename
identity interpretation still needs confirmation before creating splits.
No train/validation/test split is created by this command. The student ran
it on Rangpur on 15 October: 211 image files, 211 mask files, 211 paired
scans, 38 candidate patients and issues=0. The report was saved to
runs/inventory.json. This verifies the filename inventory, not every scan
content or the biological identity mapping. Synthetic file-tree checks also
covered repeat-patient grouping, JSON output and these failure cases.

## Reproducible patient split preparation

Create one new manifest on a compute node:

```bash
python dataset.py --split runs/split_seed42.json
```

Defaults are seed 42, six validation patients and six test patients; with 38
candidate patient groups, 26 remain for training. These are simple project
choices, not a course-prescribed ratio or tuned result. Patient groups are
sorted before a seeded shuffle. All scans and derived slices for one prefix
stay together; both models must reuse the saved manifest. Do not choose a
new seed after observing test scores. The command refuses to overwrite an
existing file and records a hash of the case/patient inventory identities.
The real command completed on 15 October, seed 42, with patient_overlap=False:

| Split | Candidate patients | Scans |
| --- | --- | --- |
| Training | 26 | 136 |
| Validation | 6 | 48 |
| Test | 6 | 27 |
| Total | 38 | 211 |

The manifest runs/split_seed42.json was downloaded privately and inspected
locally: 38 unique disjoint patient groups, 211 unique cases, consistent
patient/case/file names and a matching recomputed identity hash. This checks
manifest consistency, not biological identity or all file contents. Confirm the filename-derived identity rule before relying on this split
for final unseen-patient claims. This command does not train a model.

Synthetic checks passed for 38 groups with two scans each: 26/6/6 allocation,
complete/disjoint grouping, reproducibility under reordered inventory,
invalid-count/invalid-inventory rejection and refusing overwrite. The real
split was created successfully on the new CPU allocation 662383/vcpu-2; the
student screenshot confirms the totals and no prefix-group overlap. The downloaded JSON was independently checked locally as described above.
The subsequent full-volume integrity audit completed with211passes and0failures;
its per-case report has been independently inspected locally.

## Full data-integrity audit

After filename inventory and split preparation, run on a CPU compute node:

```bash
python dataset.py --audit-all runs/audit_all.json
```

This reloads the full inventory and sequentially reads every MRI and mask.
It checks readable volumes, 3D shape, image/mask shape and affine agreement,
finite positive matching voxel spacing, finite MRI intensities, and finite
nonnegative integer mask codes restricted to the publisher's 0–5 mapping.
Per-code mask counts are included in the report. These checks do not prove
expert annotations are correct or replace visual inspection. No fitted
normalization, model selection, resampling or source-file edits occur.

Progress and JSON checkpoints are saved every ten cases and on failures.
The report records expected/checked/passed/failed counts, complete status,
per-case observations and the case/patient identity hash. Failures are kept
for inspection; no case is automatically dropped and the split is unchanged.
A report with complete=false is partial even if failures=0. A completed
report with any failures yields a nonzero exit status.

Local synthetic checks passed for a valid pair, nonfinite MRI values,
shifted mask affine, unknown class code and an unreadable scan, including
continued reporting, CLI exit status and unchanged source-file hashes.
The student ran the real audit on CPU job662383/vcpu-2. The screenshot
confirms Audit complete=True,checked_pairs=211,passed_pairs=211,failed_pairs=0
and the saved runs/audit_all.json. The downloaded report was independently
checked locally: all 211 case records pass their eight checks, coverage equals
the split case set and the identity hash matches. No failed
case was removed; source data and the existing split remain unchanged.

### Observed dimensions and class coverage

Of the 211 volumes, 210 are 256 x 256 x 128; training case K019_Week1 is
256 x 256 x 144.
All are stored int32 with LPS orientation, and all masks contain codes 0–5.
Image/mask spacing agrees within every pair, but differs between patients.
Training in-plane spacing ranges from 1.40625 to 1.875 mm, with through-plane
spacing approximately 1.56 mm. Preprocessing must accommodate the depth variation and
preserve image/mask alignment; the longer case is retained.

Training-only voxel counts show prostate code 5 occupies approximately 0.104%
of training voxels. This motivates per-class overlap assessment and the
planned lecture-based soft Dice pilot; no model result is claimed. We do not
fit normalization or alter the split from test-case contents. The pilot input
grid and preprocessing are defined below; their real-data effect on small
structures remains to be tested.

## Aligned training Dataset and pilot grid

HipMRIVolumeDataset reads the saved manifest, recomputes its identity hash,
checks disjoint patient/case membership and verifies each recorded filename.
It opens only the selected split's images. An optional data_root relocates
the recorded basenames without changing patient assignments. Both model
routes will use this Dataset rather than independently splitting slices.

Preprocessing follows Appendix B's per-volume z-score idea. The full MRI is
normalized by its own mean and population standard deviation, with a small
denominator clamp for constant images. There is no mask-based normalization
and no pooled validation/test statistic fitted to training. Reorientation
uses the NIfTI affine to convert both arrays from their stored orientation
to RAS. NIfTI XYZ is then transposed to PyTorch ZYX (depth, height, width).

The initial **pilot**, not a tuned final setting, resizes the complete field
of view to D,H,W = 64,128,128. This reduces voxel count by about eightfold
for the usual scans and includes the longer depth-144 case without clipping.
It is a fixed voxel-count grid, **not** a common physical-spacing grid.
Per-case spacing still differs; effective output affines and original
affines/shapes are returned. This trade-off may lose small boundary features
and must be assessed on training/validation data before final evaluation.

Images use trilinear interpolation with align_corners=False; labels use
nearest-exact interpolation on the same voxel-centre grid, so no fractional
anatomical codes are introduced. See [PyTorch interpolation documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.interpolate.html).
Each item contains image 1xDxHxW, integer mask DxHxW and one-hot target
6xDxHxW, with six fixed channels even if some labels are absent. The model
receives only image; mask/target are the supervised answers. An optional
paired left/right flip is restricted to training and disabled for this pilot.
volume_to_slices produces Dx1xHxW and Dx6xHxW for the 2D baseline. Evaluation
must preserve slice order and reassemble a whole volume before scoring.

On a compute allocation, reuse the existing manifest:

```bash
python dataset.py --loader-check runs/loader_check.json --samples 3
```

Or submit the saved CPU job from login0 (mkdir runs before submission):

```bash
mkdir -p runs
sbatch loader_check.slurm
```

This loads the first and longest training cases, plus another case if
available, checks finite tensors, one-hot labels, slice reconstruction and
retention of all six labels. Its JSON reports post-resize class counts.
LOADER_CHECK_PASSED is required from the real run; a missing label is a
failure to investigate, not permission to discard the case. The diagnostic
does not train a model or prove boundary quality. It needs CPU resources,
while subsequent model training uses the GPU.

Local synthetic checks cover an analytic LPS coordinate ramp and known
label bands, world-coordinate mapping, variable depth, constant images,
absent classes, invalid labels/nonfinite MRI, affine mismatch, unchanged
source hashes, manifest tampering/leakage, training-only file access,
paired flip and DataLoader collation. The existing real 211-case manifest
passes the new identity validation. The student subsequently supplied the
real output of CPU job 699944: B006_Week0, K019_Week1 and B006_Week1 each
passed, with image 1x64x128x128, target 6x64x128x128 and all codes 0-5
retained. LOADER_CHECK_PASSED was printed and the displayed .err was empty.
The detailed JSON report has not yet been downloaded for independent count
inspection. This three-case pilot is not a full preprocessing audit or an
accuracy test; base/apex boundary quality still needs visual evaluation.

## Initial 2D U-Net baseline

modules.py implements the standard encoder/decoder U-Net pattern with four
max-pool stages, two padded 3x3/ReLU convolutions per stage, transposed
convolution upsampling, concatenated skip features, and a 1x1 six-class head.
The initial width is 16 (16,32,64,128,256), reduced from the original paper's
64 for this pilot. Padding keeps output size equal to input rather than
the paper's valid-convolution cropping. H,W must be multiples of 16.
There are no pretrained weights or extra feature extractors.

The model returns logits. SoftDiceLoss applies softmax once, computes
(2*intersection+1e-6)/(predicted_mass+target_mass+1e-6) separately for each
class over batch/spatial axes, and returns one minus the equally weighted
class mean. Background is included by default. Finite smoothing handles
empty classes; this training loss is not a substitute for per-case hard
Dice/IoU evaluation. Adam and the explicit lecture training loop are now
implemented in train.py. The CAN3D paper's specialised DSF loss is not used here.

```bash
python -m unittest -v test_dataset test_modules test_experiment
```

All 17 local synthetic tests passed on 26 October. U-Net preserves expected
shape and produces finite nonzero head gradients; Dice is checked against
a hand-calculated uniform case and near-perfect/disjoint predictions.
These are implementation checks, not trained segmentation results.
Local validation: Python 3.12.14, PyTorch 2.14.0, nibabel 5.4.2, NumPy 2.5.3.
Remote CUDA diagnostic: PyTorch 2.14.0+cu130, CUDA package 13.0, A100 40GB.
Record complete remote dependency versions with the eventual training run.

## Baseline training, checkpoints and prediction

train.py now supports three explicitly separated modes. This driver currently
supports **UNet2D only**; CAN3D still requires its own verified architecture
and volumetric training route. No real-data model has been trained yet.

1. **debug** fits up to three prostate-bearing slices and one empty-prostate
   slice from the first training case. Labels select these slices only for
   training-only debugging, not final training or evaluation. It runs Adam,
   checks finite loss/gradients, verifies parameter changes, saves debug.pt,
   reloads it and compares outputs. BASELINE_DEBUG_PASSED requires decreased
   selected-training loss and successful weight reload; it does not require
   a clinical/test Dice target and is not evidence of generalization.
2. **fit** uses every slice from every training case. Cases and slices are
   shuffled using seed+epoch. Each epoch evaluates complete validation
   volumes with eval/no_grad. best.pt is selected by validation patient-macro
   prostate Dice; last.pt includes Adam state and epoch/history for resuming.
   There is no test-image access. Fixed learning rate 1e-3, base width 16,
   slice batch size 8 and 30 epochs are initial configurable choices.
3. **test** loads an already fixed fit checkpoint and evaluates every held-out
   test volume, only with explicit --final-test. A debug checkpoint is refused.
   Do not change the split, grid or settings in response to final test scores.

Run the short, training-only GPU check first. The saved Slurm script follows
Ed #75 (partition/account comp3710, one GPU) and uses a ten-minute time limit:

```bash
mkdir -p runs
sbatch baseline_debug.slurm
```

For job JOBID, it writes runs/baseline_debug_JOBID/{config.json,history.json,
debug_report.json,debug.pt,curves.png}. The script then runs predict.py on
that same training case in a new process and writes previews under
runs/baseline_preview_JOBID. Require BASELINE_DEBUG_PASSED and
PREDICTION_PREVIEW_SAVED in its .out; inspect .err as well. The real A100
run passed in job 700021, as shown in the student's 27 October screenshot.
Selected-training loss decreased from 0.903778 to 0.435347 in 100 steps;
selected-slice prostate Dice reached 0.8743. The complete B006_Week0 training-case
preview had prostate Dice 0.2814 and reported 0.2223 seconds per case.
The selected slices and complete volume have different evaluation scopes;
neither result is a validation/test score or a controlled final resource benchmark.
Parameter count was 1,940,902. Detailed saved JSON reports and Slurm accounting
have not been inspected. Submit once, then query with squeue.

The saved baseline_fit.slurm requests one GPU for a two-hour initial fit,
using all 136 training scans and the saved validation split for 30 requested
epochs. It never evaluates the test split. The wall-time budget is a pilot
limit, not an estimate or guarantee that 30 epochs will finish:

```bash
mkdir -p runs
sbatch baseline_fit.slurm
```

Inspect runs/baseline_fit_JOBID.out and .err; checkpoints, history and curves
are under runs/baseline_fit_JOBID/. Require BASELINE_FIT_COMPLETED before
claiming all requested epochs finished. If interrupted, preserve both best.pt
and last.pt for resumption into a new output directory.

Alternatively, **inside an appropriate compute allocation**:

```bash
python train.py --mode fit --device cuda --epochs 30 --output runs/baseline_fit_seed42
```

If an allocation ends after a saved epoch, resume into a new directory:

```bash
python train.py --mode fit --device cuda --epochs 30 --resume runs/baseline_fit_seed42/last.pt --output runs/baseline_fit_resume
```

Keep best.pt beside last.pt when resuming. The previous best is retained
even if later epochs do not improve it. Resume rejects changes to model,
shape, saved manifest bytes, seed, learning rate and batch size. Bitwise
reproducibility across different hardware/library versions is not assumed.
No full fit or final test is automatically started by baseline_debug.slurm.

For a validation preview, using a real fit checkpoint:

```bash
python predict.py --checkpoint runs/baseline_fit_seed42/best.pt --split validation --output runs/baseline_validation_preview
```

predict.py reloads weights, processes every slice of each selected case,
reassembles the volume and prints hard per-class overlap metrics. It saves
a resized-RAS NIfTI mask, prostate truth/prediction/FP-red/FN-blue panels
at first/largest/last prostate slices, and preview_metrics.json. A preview
subset is explicitly labelled as such; it is not a full test benchmark.
Debug checkpoints may only preview their original training case.

For final full test evaluation **after settings are fixed**, use:

```bash
python train.py --mode test --final-test --checkpoint runs/baseline_fit_seed42/best.pt --output runs/baseline_final_test
```

Hard metrics use exact whole-volume voxel counts for each of the six classes.
If truth and prediction are both empty, that class is None and excluded from
its mean; if only one is empty its overlap is zero. Reports include both
scan-macro means and patient-macro means (mean scans within a patient, then
mean patients), so frequently scanned patients do not dominate the latter.
Soft training/validation loss and hard Dice are labelled separately. Curves
plot training/validation losses and validation patient-macro prostate Dice.

Checkpoints bind to the exact manifest file hash and record model/preprocessing
settings, source-file hashes, dependency versions, seed, host and job ID.
New run directories refuse overwrite. Peak memory is PyTorch allocated
memory, not total device/context memory; debug peak is captured before
checkpoint reload. Prediction timing covers preprocessed CPU input through
complete-case CPU hard mask, including transfers/reassembly, excluding disk,
metrics and plotting. CUDA preview timing includes a warm-up batch. Treat
initial validation/debug timings as descriptive; a controlled final benchmark
against CAN3D on identical inputs/hardware remains pending.

Synthetic integration tests actually fit a tiny temporary fixture, reload
weights, export a spatially checked NIfTI, resume training, preserve the old
best checkpoint, and exercise final-test gates. Held-out image files are
deliberately absent during debug/fit tests, so accidental access fails.
Known-count Dice/IoU, empty classes and unequal patient scan counts also have
independent expected-answer checks. Synthetic plotting samples stay private
and are not presented as real MRI results.

## First full-fit failure and controlled sampling pilot

Job 703884 ran on an A100 for two hours and timed out after 25 completed
epochs. Its downloaded config, history and latest validation JSON match the
saved manifest, all 48 validation cases and the original source hashes.
Each of the 25 validation prostate Dice values was exactly zero in history.
The epoch-25 report contains:

| Class | Patient-macro validation Dice | True voxels | Predicted voxels |
|---|---:|---:|---:|
| Background | 0.9966 | 30,185,205 | 30,147,598 |
| Body | 0.9777 | 18,184,941 | 18,596,637 |
| Bones | 0.9006 | 1,663,100 | 1,587,413 |
| Bladder | 0 | 180,936 | 0 |
| Rectum | 0 | 67,549 | 0 |
| Prostate | 0 | 49,917 | 0 |

Thus decreased mean training loss did not establish usable small-organ
segmentation. The labels exist in every validation case, ruling out missing
validation prostate targets as the explanation. Class/slice imbalance is a
plausible optimization issue, not a proven sole cause. Data/preprocessing,
prediction overlays and training coverage remain relevant diagnostic checks.
These are validation findings, not held-out test results or final failure
case autopsies. The timed-out run's files are retained.

The next separate run changes **only training slice sampling**. The original
`--slice-sampling all` remains the default. Optional `prostate-balanced`
draws half of each training case's slice budget from slices with label 5
and half without; positive slices may repeat and some other slices are not
visited that epoch. It still reads all training cases and uses the same total
slice budget per epoch. Interleaving guarantees 4 positive and 4 negative
slices for the pilot's even batch size of 8 when both groups exist. If a
case has only one group, every slice is used once. Whole-volume validation
and final testing never use this sampling or mask-based input selection.

This is a documented training adaptation to investigate the failure, not a
claimed exact lecture recipe or a demonstrated improvement. Architecture,
soft Dice loss, Adam learning rate, seed, grid and patient split are retained.
Start from the same random seed in a new run; changing sampling is refused
when resuming a different sampling configuration. For later comparison with
CAN3D, disclose this 2D batch-composition choice and compare on identical
whole-volume evaluation cases with clearly stated training budgets.

```bash
mkdir -p runs
sbatch baseline_balance_pilot.slurm
```

This requests one GPU for at most 30 minutes and 3 epochs, writing logs to
runs/baseline_balance_JOBID.out/.err and reports/checkpoints under
runs/baseline_balance_JOBID/. It does not resume or overwrite job 703884 and
does not access test images. New logs show training/validation phases, epoch
duration, sampled prostate-slice counts and validation predicted voxel counts
for every class. Inspect organ predictions and validation Dice before choosing
a longer fit. Three epochs are a diagnostic budget, not sufficient evidence
of convergence or a guarantee that sampling alone solves the failure.

The student's log for balanced pilot 704488 shows all 3 epochs completed on
CUDA. Each epoch drew 4,352 prostate-positive slices out of 8,704 draws,
confirming the intended 50% sampling. Training losses were 0.588142, 0.544909
and 0.466477; epoch times were 249.6, 244.6 and 266.1 seconds (excluding some
checkpoint/plot overhead). Validation prostate Dice remained displayed as
0.000000, with exactly zero predicted prostate and bladder voxels. Predicted
rectum counts changed from 0 to 676 to 114,516 (the final count is confirmed
by the downloaded JSON; the terminal line wrapped). The downloaded config,
history and validation report match the saved manifest and all 48 validation
cases. Epoch-3 patient-macro rectum Dice was 0.5600, while bladder and prostate
Dice were exactly zero. BASELINE_FIT_COMPLETED was shown; .err has not yet
been supplied. Three epochs do not
establish convergence or demonstrate that this sampling change fixes the
prostate failure. Preserve this run and inspect predictions before extending it.

For one training and one validation case, baseline_inspect.slurm reloads
the pilot's last.pt and uses CPU inference to save diagnostic masks, prostate
truth/prediction/error panels and per-class counts. CPU is sufficient for
these two previews; their timing is not a comparable GPU benchmark. It reads
no test images and updates no weights:

```bash
sbatch baseline_inspect.slurm
```

The default checkpoint is runs/baseline_balance_704488/last.pt; an alternate
can be supplied with INSPECT_CHECKPOINT. Logs are runs/baseline_inspect_JOBID
(.out/.err); preview subfolders are runs/baseline_inspect_JOBID/train and
/validation. Wait for BASELINE_INSPECTION_COMPLETED, then inspect the actual
panels and metrics; a marker only confirms the commands completed.

Inspection job 704533 finished both previews with an empty displayed .err.
The downloaded PNGs were visually inspected and the prediction NIfTI/JSON
files were received locally. B006_Week0 (training) has 1,287 true prostate
voxels and zero predicted; C032_Week0 (validation) has 683 true and zero
predicted. Both prostate Dice values are zero; rectum Dice values are 0.5680
and 0.3710 respectively. Both previews record the same checkpoint SHA256.
The displayed labels are present in the MRI panels, with no obvious global
offset in these examples, but these previews cannot certify all anatomical
annotations or rule out every preprocessing issue. The failure on a training
example means this is not solely evidence of poor unseen-patient performance.

The next bounded diagnostic extends the same balanced model from epoch 3
to epoch 8 (five additional epochs), retaining Adam state, best checkpoint,
history, split, input grid, loss, sampling and hyperparameters. It is not a
restart and makes no claim that longer training will necessarily solve the
prostate failure. At roughly four minutes per completed pilot epoch, the
40-minute allocation is an upper budget; actual runtime remains variable.

```bash
sbatch baseline_balance_resume.slurm
```

The source last.pt and matching best.pt remain under
runs/baseline_balance_704488/. New artifacts go under
runs/baseline_balance_resume_JOBID/; logs use the same stem with .out/.err.
Review per-class predictions and validation results before another extension.
If prostate remains absent at epoch 8, revisit the optimization/loss diagnosis
instead of automatically requesting another long fit. No test cases are used.

All 20 local synthetic tests passed, including rare-positive batch composition,
reproducibility, single-group fallback, real optimization/checkpoint resume,
refusal to change sampling on resume and held-out access guards. The new
sampling functionality has real confirmation in the pilot above; improved
prostate segmentation has not been demonstrated.

## Planned training and comparison

Use readable PyTorch modules and the teacher's training sequence:
model.train -> batch to device -> forward -> loss -> optimizer.zero_grad ->
loss.backward -> optimizer.step. Validation/prediction use model.eval and
torch.no_grad with no weight updates.

The initial controlled pilot will use Adam and per-class soft Dice loss for
both models, following the segmentation lectures. Learning rate 1e-3 is a
candidate from the course CNN example, not a verified MRI optimum. Determine
batch size, epochs, channel widths and input grid using a small allocated-GPU
smoke test and validation evidence. Do not perform training on login0.

The CAN3D paper uses a specialised Dice Squared Focal loss and a learning-rate
schedule. Our initial lecture-based Dice/fixed-rate pilot is an explicit
training adaptation, not a full reproduction of its experimental recipe.
Any later change must be source-grounded, understood and compared on
validation data; do not automatically add unrelated advanced methods.

Keep patient split, preprocessing, loss and evaluation procedures comparable;
record any model-specific tuning and compute budgets. Save the best validation
prostate Dice checkpoint and use test patients only after choices are fixed.

## Evaluation and required outputs

- Report per-class Dice and IoU, explicitly including prostate; distinguish
  Dice score from Dice loss. State aggregation, smoothing and empty-mask rules.
- CAN3D's task-table target is Dice >=0.7; confirm with the tutor which class
  aggregation is intended. Do not silently treat background accuracy as Dice.
- Profile parameter count, peak GPU VRAM and full-case inference time on the
  same hardware/input grid, including the full slice stack for the 2D baseline.
- Inspect 3-5 actual failures: input, truth, prediction and false-positive/
  false-negative overlays; discuss base/apex, inter-slice continuity and leakage.
- Give a measured engineering recommendation and limitations. No clinical
  performance or resource advantage is claimed before experiments.

## Planned implementation

| File | Responsibility |
| --- | --- |
| modules.py | PyTorch CAN3D and standard 2D U-Net components |
| dataset.py | NIfTI audit/loading, patient splits, aligned preprocessing and slice views |
| train.py | Training, validation, final evaluation, curves and checkpoint saving |
| predict.py | Load saved weights, volume/slice inference and error visualisations |

dataset.py currently implements filename-pair inventory, candidate patient
indexing, grouped splitting, one-pair and full data-integrity auditing, and
a publisher-labelled visual preview. Full-audit code is locally checked;
its real 211-pair run completed successfully and its report was inspected
locally. The real three-case preprocessing pilot also passed.
The real filename inventory and full content audit have passed. Biological
identity grouping still uses documented filename prefixes. Preprocessing,
the training Dataset and baseline modules.py have synthetic checks; train.py
and predict.py now support the tested baseline workflow. CAN3D and real-data
baseline training remain pending. Initial remote imports are verified:
PyTorch 2.14.0+cu130 and nibabel 5.4.2 on the CPU allocation. Full dependency
versions and runnable training commands remain to be recorded. CUDA execution
passed the tiny check described below. Data, weights and private audit files stay out of Git.
Existing folder name HipMRI_CAN_s4919055 is retained; it does not determine
model dimensionality.

## GPU diagnostic preparation

cuda_check.py and cuda_check.slurm implement a synthetic CUDA check, separate
from segmentation training. The batch directives follow the updated course
[Ed #6](https://edstem.org/au/courses/38242/discussion/3476306) and
[Ed #75](https://edstem.org/au/courses/38242/discussion/3537106) configuration:
partition/account comp3710, one GPU and a five-minute limit, without the
older guide's explicit memory request.

After uploading both files, submit from the remote project directory on the
login node:

```bash
mkdir -p runs
sbatch cuda_check.slurm
squeue --me
```

The script reuses the existing torch environment. Logs are written under
runs/cuda_check_JOBID.out and .err, where Slurm substitutes the job number.
It prints CUDA/device information, then checks a synthetic16-cubed Conv3d
forward pass and finite parameter gradients after backward. It reads no MRI,
updates no model weights and does not estimate CAN3D accuracy or full-model
memory requirements. Require CUDA_CHECK_PASSED in the actual job output.
Local script/shell syntax and rejection of missing Slurm allocation or CUDA
were checked. Real job 696947 passed on a100-8, as shown in the output
supplied by the student on 26 October. It reported CUDA=True, A100-PCIE-40GB,
output shape 1x2x16x16x16, finite backward gradients and CUDA_CHECK_PASSED.
Reported peak allocation was 0.173828125 MiB for this tiny diagnostic only.
The .err log and Slurm accounting were not supplied; no full-model resource
or training-performance claim is inferred from this diagnostic.

## Feasibility review and check-off

The student plans to attend on **27 October 2026**; completion has not yet
been recorded. The following is the written review draft, prepared 26 October.
Its proposed thresholds and compute limits are project choices, not measured
results or guarantees. The student needs to read and understand it before
discussing it with the tutor.

### Written feasibility review

**1. User Need, Scope & Acceptance Criteria**

Build an MRI contour-drafting prototype for an imaging specialist to review.
Compare whether CAN3D improves prostate boundaries enough to justify its cost over 2D U-Net.

Proposed acceptance targets:
- No overlap between patient groups across splits.
- Prostate test Dice at least 0.70; report per-class Dice and IoU.
- Inspect 3-5 real boundary failures with prediction/error overlays.
- Keep the 3D pilot below 35 GiB peak allocated memory on a 40 GB A100.
- Compare parameter count and inference time per complete case.

**2. Model Choice & Course Concepts**

- Model: planned CAN3D (Hard); implemented 2D U-Net baseline.
- Data: HipMRI, 211 scans and 38 filename-derived patient groups; train/validation/test = 26/6/6 groups.
- Shared preprocessing: z-score, aligned 64x128x128 pilot volumes and six-class masks; prostate is label 5.
- Course methods: convolution, dilation, skip connections, Adam and soft Dice. The training recipe is adapted to the course.

**3. Preliminary Feasibility Evidence**

- All 211 image/mask pairs passed integrity checks; three real training cases passed the resized loader check.
- A tiny A100 CUDA forward/backward test passed; baseline code passed 17 local synthetic tests.
- GPU debug 700021 passed; the first full-fit pilot timed out after 25 epochs and missed small organs on validation. A sampling diagnostic and CAN3D remain pending.

**4. Risks, Budget & Fallback**

- Risks: a small prostate class, loss of boundary detail during resizing and GPU queues.
- Next: diagnose the failed small-organ predictions, test balanced training slices, then implement CAN3D.
- Budget: one GPU; ten-minute debug allocation, then two-hour fit pilots; eight slices per 2D batch or one volume per 3D batch.
- Fallback: retain the tested 2D baseline. Adjust settings with validation data; use the test split only after choices are fixed.

**Code and AI disclosure.** Codex assisted with course- and paper-based PyTorch code, tests and this draft.
I ran the real-data checks and remain responsible for understanding and verifying the work.

## Artificial Intelligence Usage Disclosure

Codex assisted with reading the specification and published lecture PDFs,
checking selected recording captions/frames and course code, locating papers
and data sources, inspecting an incorrect local brain archive, explaining Git
and Rangpur steps, and drafting/revising project documentation. On 15 October
Codex checked the latest Blackboard sheet and revised the plan to Hard CAN3D
at the student's request. Codex wrote the initial one-pair audit, which the
student ran on real data and supplied as a screenshot. Codex then organised
it into dataset.py and checked aligned/mismatched/fractional-label synthetic
NIfTI examples locally. The student uploaded and successfully ran the saved
file on the same real B006_Week0 pair; all four checks passed. Codex added an
optional saved PNG preview and checked it on synthetic LPS data locally;
real-data B006_Week0 preview generation subsequently succeeded on Rangpur;
selected-slice visual inspection was completed. Codex retrieved the publisher
label-description file and added its explicit mapping and citation. Codex added filename-pair inventory and tested it on synthetic
file trees, including duplicates and missing pairs. The student subsequently
ran the real inventory on Rangpur: 211 pairs, 38 candidate patient IDs and
zero reported issues. The student has not independently explained or
implemented that logic yet. Codex added seeded patient-group splitting and
checked its correctness and overwrite protection locally. The student
subsequently generated the real split on Rangpur: train26/136 scans,
validation6/48, test6/27, with no prefix-group overlap. The downloaded manifest was independently inspected locally; groups/cases
and the saved identity hash are internally consistent. Codex added full-volume integrity auditing and checked it on synthetic valid
and damaged data locally. The student then ran the real211pair audit on
Rangpur, reporting211passes and0failures. Codex prepared a source-grounded
Slurm GPU diagnostic and checked local syntax/negative paths; the student
subsequently supplied its passing A100 output for job 696947. On 26 October
Codex added manifest validation, aligned preprocessing, the PyTorch Dataset,
slice conversion, training-only paired augmentation, a saved loader-check
job, the initial 2D U-Net and lecture-based soft Dice loss. Codex checked
13 synthetic data/model tests and revalidated the real saved manifest.
The student subsequently supplied the successful real three-case loader
output for job 699944. Codex added the baseline Adam/debug/fit/test driver,
checkpoint persistence/resume, metrics, reload/prediction and plots. The
expanded 17-test suite includes actual synthetic training and safe held-out
access tests. The student has not independently explained or implemented
these additions. On 27 October the student supplied the passing real MRI
GPU debug/prediction output for job 700021. Codex recorded its limited
training-case scope and added a two-hour baseline fit Slurm script, checked
with bash -n locally. The student ran the full-fit pilot and downloaded its
JSON reports after the two-hour timeout. Codex checked all 25 history entries,
per-class true/predicted counts, validation identities and source/manifest
hashes, then added an optional balanced training-slice sampler and a separate
three-epoch GPU pilot. All 20 synthetic tests passed. This sampling change
has not yet been validated on real MRI; CAN3D remains unimplemented.
Future entries will name actual assistance and verification, without claiming it in advance.

## Data acknowledgement

Data were provided by HipMRI Study. The shared data_use_agreement.txt,
read in the student's terminal on 15 October, requests acknowledgement and
citation, specifies non-commercial use, and excludes attempts to identify or
contact participants. This is an educational prototype. We cite the original
collection and the acquisition/treatment-planning publication linked in that
file: [Dowling et al. (2015)](https://doi.org/10.1016/j.ijrobp.2015.08.045).

## References

- COMP3710 final-report sheet v2.5, retrieved from the official Ed announcement
  on 26 October 2026, especially Sections 2.2-6 and Appendices A-B.
- Dai, W. et al. (2021), [CAN3D: Fast 3D Medical Image Segmentation via Compact
  Context Aggregation](https://arxiv.org/abs/2109.05443), main reference [5].
- Ronneberger, O., Fischer, P. and Brox, T. (2015), [U-Net: Convolutional
  Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597),
  baseline reference [1].
- Contact_6_Neural_Networks.pdf; Contact_7_ConvNets_Segmentation_Object_Detection.pdf.
- Course [MNIST CNN notebook](https://colab.research.google.com/drive/138ORmJkYvktCRmU8TIfsin84R_p4A2iG?usp=sharing).
- Dowling, J. and Greer, P. (2021), [Labelled weekly MR images of the male
  pelvis](https://doi.org/10.25919/45t8-p065), CSIRO Data Collection, v2.

## Submission checklist

- [ ] Confirm date and complete compulsory feasibility check-off.
- [ ] Audit real 3D images/masks and patient identity; save split manifest.
- [ ] Implement and verify the five required files and baseline comparison.
- [ ] Complete training, final test, resource profiling and failure analysis.
- [ ] Keep meaningful commits at genuine milestones.
- [ ] Finish README with reproducible commands, feasibility review and AI disclosure.
- [ ] Open working upstream PR to topic-recognition before the actual deadline.
- [ ] Submit README PDF via Turnitin and respond to later teaching-team review.
