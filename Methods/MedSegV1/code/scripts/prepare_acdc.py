import argparse
import csv
import json
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image


CLASS_NAMES = {
    0: "background",
    1: "right_ventricle",
    2: "myocardium",
    3: "left_ventricle",
}


def _resampling_modes():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BILINEAR, Image.Resampling.NEAREST
    return Image.BILINEAR, Image.NEAREST


BILINEAR, NEAREST = _resampling_modes()


def find_acdc_training_root(raw_root):
    root = Path(raw_root).expanduser().resolve()
    candidates = [
        root,
        root / "database",
        root / "ACDC" / "database",
        root / "data" / "ACDC" / "database",
    ]
    for candidate in candidates:
        training = candidate / "training"
        if training.is_dir() and any(training.glob("patient*/Info.cfg")):
            return training
    raise FileNotFoundError(f"Could not find ACDC database/training under {root}")


def read_patient_list(path):
    patients = []
    seen = set()
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        patient = line.split("_frame", 1)[0]
        if patient not in seen:
            patients.append(patient)
            seen.add(patient)
    return patients


def _resolve_split_file(root, split_name, preferred_name, legacy_name):
    preferred = root / preferred_name
    legacy = root / legacy_name
    if preferred.is_file():
        return preferred
    if legacy.is_file():
        return legacy
    raise FileNotFoundError(
        f"Missing ACDC {split_name} split manifest under {root}. Expected "
        f"release manifest '{preferred_name}' or legacy manifest '{legacy_name}'."
    )


def load_split(split_root):
    root = Path(split_root).expanduser().resolve()
    split_files = {
        "training": _resolve_split_file(
            root, "training", "train_subjects.txt", "train_patients.txt"
        ),
        "validation": _resolve_split_file(
            root, "validation", "val_subjects.txt", "val_patients.txt"
        ),
        "testing": _resolve_split_file(
            root, "testing", "test_subjects.txt", "test_patients.txt"
        ),
    }
    split = {
        split_name: read_patient_list(path)
        for split_name, path in split_files.items()
    }
    all_patients = [patient for patients in split.values() for patient in patients]
    if len(all_patients) != len(set(all_patients)):
        raise ValueError(f"Duplicate patients found in split root: {root}")
    return split


def normalize_mri_frame_to_uint8(volume, lower, upper):
    volume = np.asarray(volume, dtype=np.float32)
    finite = volume[np.isfinite(volume)]
    foreground = finite[finite > 0]
    reference = foreground if foreground.size else finite
    if reference.size == 0:
        return np.zeros_like(volume, dtype=np.uint8)
    lo, hi = np.percentile(reference, [lower, upper])
    if hi <= lo:
        return np.zeros_like(volume, dtype=np.uint8)
    volume = np.clip(volume, lo, hi)
    volume = (volume - lo) / (hi - lo)
    return np.round(volume * 255.0).astype(np.uint8)


def resize_image(slice_2d, output_size):
    image = Image.fromarray(slice_2d.astype(np.uint8))
    image = image.resize((output_size, output_size), resample=BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def resize_label(slice_2d, output_size):
    image = Image.fromarray(slice_2d.astype(np.uint8))
    image = image.resize((output_size, output_size), resample=NEAREST)
    return np.asarray(image, dtype=np.uint8)


def foreground_rgb_from_label(label, num_classes):
    if int(num_classes) != 4:
        raise ValueError("ACDC foreground RGB encoding expects num_classes=4")
    label = np.asarray(label, dtype=np.uint8)
    mask_rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    for class_id in range(1, int(num_classes)):
        mask_rgb[..., class_id - 1] = np.where(label == class_id, 255, 0).astype(np.uint8)
    return mask_rgb


def frame_image_paths(patient_dir):
    paths = []
    for path in sorted(patient_dir.glob(f"{patient_dir.name}_frame*.nii.gz")):
        if path.name.endswith("_gt.nii.gz"):
            continue
        paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No labelled ACDC frame images found in {patient_dir}")
    return paths


def frame_id(path):
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def reset_output_dir(out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for child in out_dir.iterdir():
        if child.is_dir() and child.name in {"training", "validation", "testing", "train", "val", "test"}:
            shutil.rmtree(child)
        elif child.is_file() and child.name in {"manifest.csv", "summary.json"}:
            child.unlink()


def preprocess_frame(image_path, label_path, split, out_dir, args, rows):
    image_volume = nib.load(str(image_path)).get_fdata()
    label_volume = nib.load(str(label_path)).get_fdata()
    if image_volume.shape != label_volume.shape:
        raise ValueError(
            f"Shape mismatch for {image_path.name}: {image_volume.shape} vs {label_volume.shape}"
        )

    labels = np.rint(label_volume).astype(np.uint8)
    if labels.max() >= args.num_classes:
        raise ValueError(
            f"Unexpected ACDC label value {int(labels.max())} in {label_path}; "
            f"num_classes={args.num_classes}"
        )
    image_volume = normalize_mri_frame_to_uint8(
        image_volume,
        args.clip_lower,
        args.clip_upper,
    )
    fid = frame_id(image_path)
    image_dir = out_dir / split / "images"
    mask_dir = out_dir / split / "masks"
    label_map_dir = out_dir / split / "label_maps"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    label_map_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for z_index in range(image_volume.shape[2]):
        slice_id = f"{fid}_z{z_index:03d}"
        image_slice = resize_image(image_volume[:, :, z_index], args.output_size)
        image_rgb = np.repeat(image_slice[:, :, None], 3, axis=2)
        label_slice = resize_label(labels[:, :, z_index], args.output_size)
        mask_rgb = foreground_rgb_from_label(label_slice, args.num_classes)

        image_out = image_dir / f"{slice_id}.png"
        mask_out = mask_dir / f"{slice_id}.png"
        label_map_out = label_map_dir / f"{slice_id}.png"
        Image.fromarray(image_rgb, mode="RGB").save(image_out)
        Image.fromarray(mask_rgb, mode="RGB").save(mask_out)
        Image.fromarray(label_slice.astype(np.uint8), mode="L").save(label_map_out)

        class_pixels = {
            f"class{class_id}_pixels": int((label_slice == class_id).sum())
            for class_id in range(1, args.num_classes)
        }
        rows.append(
            {
                "split": split,
                "patient_id": image_path.parent.name,
                "frame_id": fid,
                "z_index": z_index,
                "slice_id": slice_id,
                "image_file": str(image_out.relative_to(out_dir)),
                "mask_file": str(mask_out.relative_to(out_dir)),
                "label_map_file": str(label_map_out.relative_to(out_dir)),
                "image_path": str(image_path),
                "label_path": str(label_path),
                "empty_foreground": int(sum(class_pixels.values()) == 0),
                **class_pixels,
            }
        )
        written += 1
    return written


def summarize_rows(rows, num_classes):
    summary = {}
    for split in ("training", "validation", "testing"):
        split_rows = [row for row in rows if row["split"] == split]
        summary[split] = {
            "slices": len(split_rows),
            "empty_foreground_slices": sum(int(row["empty_foreground"]) for row in split_rows),
        }
        for class_id in range(1, num_classes):
            key = f"class{class_id}_pixels"
            summary[split][key] = sum(int(row[key]) for row in split_rows)
            summary[split][f"class{class_id}_nonempty_slices"] = sum(
                int(row[key]) > 0 for row in split_rows
            )
    return summary


def main():
    args = create_argparser().parse_args()
    if int(args.num_classes) != 4:
        raise ValueError("ACDC preprocessing expects --num_classes 4")
    training_root = find_acdc_training_root(args.raw_root)
    split = load_split(args.split_root)
    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.clean:
        reset_output_dir(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.max_patients_per_split > 0:
        split = {
            split_name: patients[: args.max_patients_per_split]
            for split_name, patients in split.items()
        }

    rows = []
    patient_counts = {key: len(value) for key, value in split.items()}
    slice_counts = {key: 0 for key in split}
    for split_name, patient_ids in split.items():
        for patient_index, patient_id in enumerate(patient_ids, 1):
            patient_dir = training_root / patient_id
            if not patient_dir.is_dir():
                raise FileNotFoundError(f"Missing ACDC patient directory: {patient_dir}")
            for image_path in frame_image_paths(patient_dir):
                label_path = image_path.with_name(f"{frame_id(image_path)}_gt.nii.gz")
                if not label_path.exists():
                    raise FileNotFoundError(f"Missing ACDC GT for {image_path}: {label_path}")
                slice_counts[split_name] += preprocess_frame(
                    image_path,
                    label_path,
                    split_name,
                    out_dir,
                    args,
                    rows,
                )
            print(
                f"{split_name}: processed {patient_index}/{len(patient_ids)} "
                f"{patient_id} -> {slice_counts[split_name]} slices so far"
            )

    fieldnames = [
        "split",
        "patient_id",
        "frame_id",
        "z_index",
        "slice_id",
        "image_file",
        "mask_file",
        "label_map_file",
        "image_path",
        "label_path",
        "empty_foreground",
    ] + [f"class{class_id}_pixels" for class_id in range(1, args.num_classes)]
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "raw_training_root": str(training_root),
        "split_root": str(Path(args.split_root).expanduser().resolve()),
        "out_dir": str(out_dir),
        "split": "cascade_70_10_20",
        "patient_counts": patient_counts,
        "slice_counts": slice_counts,
        "split_summaries": summarize_rows(rows, args.num_classes),
        "total_slices": len(rows),
        "output_size": args.output_size,
        "intensity_clip_percentiles": [args.clip_lower, args.clip_upper],
        "normalization": "per labelled ED/ES frame, nonzero percentile clip, scaled to uint8 PNG",
        "num_classes": args.num_classes,
        "num_mask_channels": args.num_classes - 1,
        "class_names": CLASS_NAMES,
        "manifest": str(manifest_path),
        "storage_layout": "<split>/images/*.png, <split>/masks/*.png, <split>/label_maps/*.png",
        "label_policy": (
            "ACDC labels are preserved as RGB foreground channels: "
            "R=RV, G=myocardium, B=LV; all-zero foreground channels are background."
        ),
        "metric_policy": (
            "Foreground Dice/IoU should aggregate classes 1..3 and skip empty-GT "
            "slice/class observations; background class 0 is excluded."
        ),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_root",
        default="../../data/ACDC/database",
        help="Shared raw ACDC database root or a parent containing database/training.",
    )
    parser.add_argument(
        "--split_root",
        default="../../data/ACDC/splits/cascade_70_10_20",
        help=(
            "Directory containing release train_subjects.txt, val_subjects.txt, and "
            "test_subjects.txt manifests, or the legacy *_patients.txt equivalents."
        ),
    )
    parser.add_argument(
        "--out_dir",
        default="../data_preprocessed/acdc_medsegv1_multiclass_png",
        help="Project-local MedSegV1 ACDC preprocessed multi-class PNG slices.",
    )
    parser.add_argument("--output_size", type=int, default=256)
    parser.add_argument("--clip_lower", type=float, default=1.0)
    parser.add_argument("--clip_upper", type=float, default=99.0)
    parser.add_argument("--num_classes", type=int, default=4)
    parser.add_argument("--max_patients_per_split", type=int, default=0)
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clean generated ACDC split directories before writing the cache.",
    )
    return parser


if __name__ == "__main__":
    main()
