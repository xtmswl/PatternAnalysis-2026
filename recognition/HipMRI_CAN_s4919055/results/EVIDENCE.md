# Evidence index for the working report

The report's validation values are summarized in validation_summary.csv, and the curves in learning_history.csv.

The later [709041 continuation update](continuation_709041.md) has independently verified history/latest-validation and inspected best-checkpoint previews. Its full per-class report is epoch 8; the epoch-7 best score is recorded in history, pending a separate best-checkpoint validation recheck.

- 703884: inspected downloaded config/history/latest validation; 25 completed epochs, then TIMEOUT from student-supplied scheduler/error output.
- 704488: inspected downloaded config/history/latest validation; 3 epochs, same fixed validation cases and manifest hash.
- 708618: inspected downloaded config/history/latest validation; matched three-epoch Dice+CE pilot. All four source hashes and fixed B/D settings checked; patient-macro scores independently recomputed.
- 708234: epochs4-8 and completion/error status from student terminal screenshots; detailed JSON pending. Blank validation-loss CSV fields mean not supplied, not zero.
- 704533: inspected two downloaded preview JSONs, PNGs and resized-RAS NIfTI outputs; independent voxel counts match reports. CPU timings are diagnostic.
- 700021: selected-slice training/debug and full-case preview values transcribed from user-supplied logs; not a validation/test benchmark.

Raw manifests, reports, MRI volumes and model weights are not copied into this results folder. The four figures are report visuals, not a redistributed training dataset. No final test results are available. The original check-off preparation content is archived separately and not rewritten as later progress.
