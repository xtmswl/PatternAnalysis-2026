# Evidence index for the working report

The report's validation values are summarized in validation_summary.csv, and the curves in learning_history.csv.

- 703884: inspected downloaded config/history/latest validation; 25 completed epochs, then TIMEOUT from student-supplied scheduler/error output.
- 704488: inspected downloaded config/history/latest validation; 3 epochs, same fixed validation cases and manifest hash.
- 708618: inspected downloaded config/history/latest validation; matched three-epoch Dice+CE pilot. All four source hashes and fixed B/D settings checked; patient-macro scores independently recomputed.
- 709556: CAN3D five-step training-only feasibility result from a supplied terminal screenshot, no displayed errors; raw smoke JSON pending collection with the first full-dataset trial. No accuracy/epoch-time claim.
- 709041: independently checked downloaded source/manifest/history/latest epoch-8 validation and best-checkpoint previews; epochs1-3 exactly retain708618 history.
- 709282: fixed epoch-7 checkpoint provenance and full48-case metrics verified; independent patient aggregation and three distinct-patient panels inspected.
- 708234: epochs4-8 and completion/error status from student terminal screenshots; detailed JSON pending. Blank validation-loss CSV fields mean not supplied, not zero.
- 704533: inspected two downloaded preview JSONs, PNGs and resized-RAS NIfTI outputs; independent voxel counts match reports. CPU timings are diagnostic.
- 700021: selected-slice training/debug and full-case preview values transcribed from user-supplied logs; not a validation/test benchmark.

Raw manifests, reports, MRI volumes and model weights are not copied into this results folder. The figures are report visuals, not a redistributed training dataset. No final test results are available. The original check-off preparation content is archived separately and not rewritten as later progress.

- 709710: downloaded current source hashes, fixed split, all class metrics and two PNG/NIfTI previews independently verified; raw709556smoke JSON also checked. Details: [CAN3D pilot](can3d_pilot_709710.md).
