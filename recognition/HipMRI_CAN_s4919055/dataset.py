"""HipMRI inventory, grouped splitting, data audits and visual previews.

Run this saved file on a Rangpur compute node, for example:
    python dataset.py --case B006_Week0

Patient IDs below are inferred from filenames and still require confirmation.
The training Dataset will follow content audit and preprocessing decisions.
"""

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_DATA_ROOT = Path("/home/groups/comp3710/HipMRI_Study_open")
# CSIRO collection v2, BriefDataDescription.txt (retrieved 15 October 2026).
LABEL_SOURCE = "https://data.csiro.au/dap/ws/v2/collections/52549/support/5034"
LABEL_NAMES = {
    0: "Background", 1: "Body", 2: "Bones", 3: "Bladder",
    4: "Rectum", 5: "Prostate",
}


def build_inventory(data_root):
    """Match image/mask filenames by case, without loading scan contents.

    B006_Week0 and B006_Week1 are grouped under candidate patient ID B006.
    Duplicate or unexpected names are reported rather than silently selected.
    This checks file pairing, not spatial metadata or labels of every volume.
    """
    root = Path(data_root)
    indexed = {}
    issues = []
    counts = {}
    for role, folder, suffix in (
        ("image", "semantic_MRs", "_LFOV"),
        ("mask", "semantic_labels_only", "_SEMANTIC"),
    ):
        directory = root / folder
        if not directory.is_dir():
            raise FileNotFoundError(f"Required data directory missing: {directory}")
        indexed[role] = {}
        counts[role] = 0
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            if path.name.endswith(".nii.gz"):
                stem = path.name[:-7]
            elif path.name.endswith(".nii"):
                stem = path.name[:-4]
            else:
                continue
            counts[role] += 1
            if not stem.endswith(suffix):
                issues.append(f"Unexpected {role} filename: {path.name}")
                continue
            case_id = stem[:-len(suffix)]
            indexed[role].setdefault(case_id, []).append(path)

    pairs = []
    cases = sorted(set(indexed["image"]) | set(indexed["mask"]))
    for case_id in cases:
        images = indexed["image"].get(case_id, [])
        masks = indexed["mask"].get(case_id, [])
        if len(images) != 1 or len(masks) != 1:
            issues.append(f"{case_id}: {len(images)} images, {len(masks)} masks; expected one each")
            continue
        match = re.fullmatch(r"([A-Za-z]+\d+)_Week(\d+)", case_id)
        if match is None:
            issues.append(f"Unrecognised patient/week naming: {case_id}")
            continue
        pairs.append({
            "case_id": case_id,
            "patient_id": match.group(1),
            "week": int(match.group(2)),
            "image_path": str(images[0].resolve()),
            "mask_path": str(masks[0].resolve()),
        })
    patient_counts = {}
    for pair in pairs:
        patient_id = pair["patient_id"]
        patient_counts[patient_id] = patient_counts.get(patient_id, 0) + 1
    if not pairs:
        issues.append("No valid image/mask pairs found")
    return {
        "data_root": str(root.resolve()),
        "image_files": counts["image"],
        "mask_files": counts["mask"],
        "paired_scans": len(pairs),
        "candidate_patients": len(patient_counts),
        "scans_per_candidate_patient": dict(sorted(patient_counts.items())),
        "patient_id_rule": "Filename prefix before _Week; identity mapping requires confirmation",
        "pairs": pairs,
        "issues": issues,
    }


def volume_metadata(volume):
    """Read shape and spatial metadata without loading image intensities."""
    return {
        "shape": tuple(volume.shape),
        "voxel_spacing": tuple(float(v) if np.isfinite(v) else None for v in volume.header.get_zooms()),
        "orientation": nib.aff2axcodes(volume.affine),
        "stored_dtype": str(volume.get_data_dtype()),
    }


def make_patient_split(inventory, seed=42, val_patients=6, test_patients=6):
    """Make one reproducible split using filename-derived patient groups.

    Counts and seed are project choices, not prescribed course settings.
    Test data must not be used to choose a more favourable seed or split.
    """
    if inventory["issues"]:
        raise ValueError("Resolve inventory issues before splitting.")
    patients = sorted({pair["patient_id"] for pair in inventory["pairs"]})
    if val_patients < 1 or test_patients < 1:
        raise ValueError("Validation and test each require at least one patient.")
    if val_patients + test_patients >= len(patients):
        raise ValueError("Patient counts must leave a nonempty training set.")
    random.Random(seed).shuffle(patients)
    groups = {
        "test": sorted(patients[:test_patients]),
        "validation": sorted(patients[test_patients:test_patients + val_patients]),
        "train": sorted(patients[test_patients + val_patients:]),
    }
    splits = {}
    for name, patient_ids in groups.items():
        selected = set(patient_ids)
        splits[name] = {
            "patient_ids": patient_ids,
            "scans": sorted(
                (pair for pair in inventory["pairs"] if pair["patient_id"] in selected),
                key=lambda pair: pair["case_id"],
            ),
        }
    overlap = any(
        set(groups[a]) & set(groups[b])
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
    )
    if overlap:
        raise ValueError("Patient overlap detected.")
    # Bind this split to the case/group list, independent of absolute paths.
    identities = sorted((p["case_id"], p["patient_id"]) for p in inventory["pairs"])
    digest = hashlib.sha256(json.dumps(identities).encode("utf-8")).hexdigest()
    return {
        "seed": seed,
        "patient_id_rule": inventory["patient_id_rule"],
        "inventory_identity_sha256": digest,
        "patient_overlap": overlap,
        "label_source": LABEL_SOURCE,
        "label_names": LABEL_NAMES,
        "splits": splits,
    }


def audit_pair(data_root, case_id):
    """Inspect one image/mask pair without changing either source file.

    Matching shape/affine is a preliminary grid check. An overlay and the
    publisher label description establish the remaining interpretation context.
    """
    data_root = Path(data_root)
    image_path = data_root / "semantic_MRs" / f"{case_id}_LFOV.nii.gz"
    mask_path = data_root / "semantic_labels_only" / f"{case_id}_SEMANTIC.nii.gz"
    return inspect_pair_files(image_path, mask_path, case_id)


def inspect_pair_files(image_path, mask_path, case_id, check_intensities=False):
    """Inspect exact inventory paths; optionally read all MRI intensities.

    This is a data-integrity check, not fitting a normalization or selecting
    model settings. Original arrays, class codes and source files are unchanged.
    """
    image = nib.load(image_path)
    mask = nib.load(mask_path)

    label_values, label_counts = np.unique(np.asanyarray(mask.dataobj), return_counts=True)
    valid_labels = bool(
        label_values.size > 0
        and np.all(np.isfinite(label_values))
        and np.all(label_values == np.round(label_values))
        and np.all(label_values >= 0)
    )
    checks = {
        "both_3d": len(image.shape) == 3 and len(mask.shape) == 3,
        "same_shape": image.shape == mask.shape,
        "same_affine": bool(
            np.allclose(image.affine, mask.affine, rtol=0, atol=1e-4)
        ),
        "finite_nonnegative_integer_labels": valid_labels,
        "known_label_codes": bool(np.all(np.isin(label_values, list(LABEL_NAMES)))),
    }
    both_3d = checks["both_3d"]
    image_spacing = np.asarray(image.header.get_zooms(), dtype=float)
    mask_spacing = np.asarray(mask.header.get_zooms(), dtype=float)
    checks["valid_voxel_spacing"] = bool(
        both_3d and np.all(np.isfinite(image_spacing))
        and np.all(np.isfinite(mask_spacing))
        and np.all(image_spacing > 0) and np.all(mask_spacing > 0)
    )
    checks["same_voxel_spacing"] = bool(
        both_3d and np.allclose(image_spacing, mask_spacing, rtol=0, atol=1e-4)
    )
    if check_intensities:
        checks["finite_image_intensities"] = bool(
            np.all(np.isfinite(np.asanyarray(image.dataobj)))
        )
    return {
        "case_id": case_id,
        "image": volume_metadata(image),
        "mask": volume_metadata(mask),
        "number_of_mask_values": int(label_values.size),
        "mask_values_preview": [v.item() if np.isfinite(v) else str(v) for v in label_values[:20]],
        "mask_value_counts": {str(v): int(n) for v, n in zip(label_values, label_counts)},
        "checks": checks,
    }


def audit_all_pairs(inventory, output_path):
    """Read each listed pair and checkpoint a report every ten cases.

    No scans are dropped, transformed or used to tune a model. A complete
    report with failures is retained so that problems can be investigated.
    """
    if inventory["issues"]:
        raise ValueError("Resolve inventory issues before the full audit.")
    pairs = inventory["pairs"]
    identities = sorted((p["case_id"], p["patient_id"]) for p in pairs)
    report = {
        "expected_pairs": len(pairs), "checked_pairs": 0,
        "passed_pairs": 0, "failed_pairs": 0, "complete": False,
        "inventory_identity_sha256": hashlib.sha256(
            json.dumps(identities).encode("utf-8")
        ).hexdigest(),
        "label_source": LABEL_SOURCE,
        "label_names": LABEL_NAMES,
        "cases": [],
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def save_report():
        # Replace only after a complete write, preserving the last checkpoint.
        temporary = output_path.with_suffix(output_path.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output_path)

    save_report()
    print(f"Reading {len(pairs)} MRI/mask pairs; source files stay unchanged.", flush=True)
    for number, pair in enumerate(pairs, start=1):
        try:
            observation = inspect_pair_files(
                pair["image_path"], pair["mask_path"], pair["case_id"],
                check_intensities=True,
            )
            passed = all(observation["checks"].values())
            observation["passed"] = passed
        except (OSError, ValueError, EOFError, np.linalg.LinAlgError,
                nib.filebasedimages.ImageFileError) as error:
            passed = False
            observation = {"case_id": pair["case_id"], "passed": False,
                           "error": f"{type(error).__name__}: {error}"}
        report["cases"].append(observation)
        report["checked_pairs"] = number
        report["passed_pairs" if passed else "failed_pairs"] += 1
        if not passed:
            print(f"FAIL: {pair['case_id']}", flush=True)
        if number % 10 == 0 or not passed or number == len(pairs):
            save_report()
            print(f"Checked {number}/{len(pairs)}; failures={report['failed_pairs']}", flush=True)
    report["complete"] = True
    save_report()
    return report


def main():
    parser = argparse.ArgumentParser(description="Inventory or audit HipMRI data.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--case", default="B006_Week0")
    parser.add_argument("--overlay", type=Path, help="Save a numeric-label preview PNG.")
    parser.add_argument("--inventory", type=Path, help="Save a full filename-pair inventory JSON.")
    parser.add_argument("--split", type=Path, help="Create a new patient-split JSON; refuses overwrite.")
    parser.add_argument("--audit-all", type=Path, help="Read all pairs and save an integrity report JSON.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-patients", type=int, default=6)
    parser.add_argument("--test-patients", type=int, default=6)
    args = parser.parse_args()

    if args.audit_all:
        if args.split or args.inventory or args.overlay:
            parser.error("Use --audit-all separately from other output modes.")
        report = audit_all_pairs(build_inventory(args.data_root), args.audit_all)
        print("Audit complete:", report["complete"])
        print("Checked pairs:", report["checked_pairs"])
        print("Passed pairs:", report["passed_pairs"])
        print("Failed pairs:", report["failed_pairs"])
        print("Report saved:", args.audit_all)
        return 1 if report["failed_pairs"] else 0

    if args.split:
        if args.inventory or args.overlay:
            parser.error("Use --split separately from --inventory and --overlay.")
        if args.split.exists():
            parser.error("Split file already exists; reuse it rather than overwriting.")
        inventory = build_inventory(args.data_root)
        split = make_patient_split(inventory, args.seed, args.val_patients, args.test_patients)
        args.split.parent.mkdir(parents=True, exist_ok=True)
        with args.split.open("x", encoding="utf-8") as stream:
            json.dump(split, stream, indent=2)
            stream.write("\n")
        for name, group in split["splits"].items():
            print(f"{name}: {len(group['patient_ids'])} patients, {len(group['scans'])} scans")
        print("patient_overlap:", split["patient_overlap"])
        print("Seed:", split["seed"])
        print("Split saved:", args.split)
        print("Both models must use this same split. Filename identity rule still needs confirmation.")
        return 0

    if args.inventory:
        if args.overlay:
            parser.error("Use --inventory separately from --overlay.")
        inventory = build_inventory(args.data_root)
        args.inventory.parent.mkdir(parents=True, exist_ok=True)
        args.inventory.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
        for key in ("image_files", "mask_files", "paired_scans", "candidate_patients"):
            print(f"{key}: {inventory[key]}")
        print("issues:", len(inventory["issues"]))
        for issue in inventory["issues"][:10]:
            print(" -", issue)
        print("Inventory saved:", args.inventory)
        print("Patient IDs are inferred from filenames; no train/test split created.")
        return 1 if inventory["issues"] else 0

    report = audit_pair(args.data_root, args.case)
    print("Case:", report["case_id"])
    for role in ("image", "mask"):
        for field, value in report[role].items():
            print(f"{role} {field}: {value}")
    print("Number of mask values:", report["number_of_mask_values"])
    print("Mask values (first 20):", report["mask_values_preview"])
    print("Label names:", LABEL_NAMES)
    for name, passed in report["checks"].items():
        print(f"{name}: {passed}")
    if args.overlay and all(report["checks"].values()):
        save_overlay(args.data_root, args.case, args.overlay)
        print("Overlay saved:", args.overlay)
    print("Next: validate the remaining volumes and patient grouping before training.")
    # A failed check stops a caller from treating this pair as ready for use.
    return 0 if all(report["checks"].values()) else 1


def save_overlay(data_root, case_id, output_path):
    """Preview each nonzero code with the publisher's anatomical label name.

    Use the axial slice with the largest area for each code. Reorient both
    volumes to RAS together; display scaling never changes the source data.
    Background code 0 is omitted from the preview.
    """
    import matplotlib

    matplotlib.use("Agg")  # Save images without a desktop on Rangpur.
    import matplotlib.pyplot as plt

    root = Path(data_root)
    image = nib.as_closest_canonical(nib.load(root / "semantic_MRs" / f"{case_id}_LFOV.nii.gz"))
    mask = nib.as_closest_canonical(nib.load(root / "semantic_labels_only" / f"{case_id}_SEMANTIC.nii.gz"))
    pixels = image.get_fdata(dtype=np.float32)
    labels = np.asanyarray(mask.dataobj)
    codes = [int(code) for code in np.unique(labels) if code != 0]
    if not codes:
        raise ValueError("No nonzero mask codes to preview.")
    finite = pixels[np.isfinite(pixels)]
    if finite.size == 0:
        raise ValueError("MRI contains no finite intensities.")
    low, high = np.percentile(finite, [1, 99])
    fig, axes = plt.subplots(len(codes), 3, figsize=(10, 3 * len(codes)), squeeze=False)
    for row, code in enumerate(codes):
        region = labels == code
        z = int(np.argmax(region.sum(axis=(0, 1))))
        section = pixels[:, :, z].T
        binary = region[:, :, z].T
        for ax in axes[row]:
            ax.set_axis_off()
        axes[row, 0].imshow(section, cmap="gray", origin="lower", vmin=low, vmax=high)
        axes[row, 0].set_title(f"{case_id}: axial RAS z={z}")
        axes[row, 1].imshow(binary, cmap="gray", origin="lower", vmin=0, vmax=1)
        axes[row, 1].set_title(f"Label {code}: {LABEL_NAMES.get(code, 'Unknown')}")
        axes[row, 2].imshow(section, cmap="gray", origin="lower", vmin=low, vmax=high)
        axes[row, 2].imshow(np.ma.masked_where(~binary, binary), cmap="autumn", origin="lower", vmin=0, vmax=1, alpha=0.45, interpolation="nearest")
        axes[row, 2].set_title(f"MRI + {LABEL_NAMES.get(code, 'Unknown')}")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
