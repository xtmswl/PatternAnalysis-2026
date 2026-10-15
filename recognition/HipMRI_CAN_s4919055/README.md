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

No project models, training results or allocated-GPU measurements exist.
The real filename inventory reports 211 MRI files, 211 mask files,
211 paired scans, 38 candidate patient IDs and zero naming/pairing issues.
The publisher label mapping is now verified. Filename-derived patient
identity interpretation remains an explicit filename-based assumption.
The full content audit completed remotely:211checked,211passed,0failed;
independent inspection of its downloaded report remains pending. The compulsory midpoint check-off is still pending.

## Research question

Does CAN3D's volumetric multi-scale context reduce prostate boundary and
slice-transition errors compared with a 2D U-Net, and what are the memory and
inference-time trade-offs? Neither model is assumed to win. Analyse both
held-out overlap scores and actual failure cases.

## Course and literature basis

The latest visible Blackboard sheet was checked on 15 October:
COMP3710_Report_2026_draft_v2.3.pdf. Its cover internally says Version 2.2
Final Assessment Specification and still displays DRAFT in the title.
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
  not directory-listing order. The full filename inventory passed; scan
  contents and any provided downsampled version still need audit.
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
independent inspection of its saved per-case report is next.

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
and the saved runs/audit_all.json. The report has not yet been downloaded
and inspected locally for per-case size/spacing/class summaries. No failed
case was removed; source data and the existing split remain unchanged.

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
its real211pair run completed successfully. Per-case report inspection
and preprocessing choices are next.
The real filename inventory and full content audit have passed. Biological
identity grouping still uses documented filename prefixes. Report inspection,
preprocessing and the training Dataset remain
pending, as do modules.py, train.py and predict.py. Initial remote imports are verified:
PyTorch 2.14.0+cu130 and nibabel 5.4.2 on the CPU allocation. Full dependency
versions and runnable training commands remain to be recorded. CUDA execution
is not yet verified. Data, weights and private audit files stay out of Git.
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
were checked. No real GPU job has been submitted or passed yet.

## Feasibility review and check-off

Check-off pending. Section 3 requires a one-page written review covering
user/scope and 3-5 measurable criteria, course/model rationale, preliminary
feasibility evidence, compute budget, risks and fallback. It is an ungraded
3-5 minute practical consultation, with no live demo/slides/oral exam required
for that consultation. Its technical write-up contributes to final documentation.

Confirm the actual date/make-up arrangements with staff now; a fully trained
final model is not required for the review, but missing this compulsory hurdle
can prevent passing the project. Prepare a real 3D data audit or allocated-GPU
smoke test before presenting the review. Do not describe planned evidence as
completed. If memory is excessive, first use justified aligned downsampling
per the task table; any change of difficulty is an explicit scope decision.

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
Slurm GPU diagnostic and checked local syntax/negative paths; actual GPU
execution remains pending.
No model has been built or trained. Future
entries will name actual assistance and verification, without claiming it in advance.

## Data acknowledgement

Data were provided by HipMRI Study. The shared data_use_agreement.txt,
read in the student's terminal on 15 October, requests acknowledgement and
citation, specifies non-commercial use, and excludes attempts to identify or
contact participants. This is an educational prototype. We cite the original
collection and the acquisition/treatment-planning publication linked in that
file: [Dowling et al. (2015)](https://doi.org/10.1016/j.ijrobp.2015.08.045).

## References

- COMP3710 Blackboard final-report sheet, downloaded 15 October 2026:
  COMP3710_Report_2026_draft_v2.3.pdf, especially Sections 2.2-6 and Appendix A.
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
