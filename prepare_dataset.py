"""Audit, preprocess, and split the Kaggle ASL Alphabet image dataset.

The source is read-only. Outputs go under data/processed and data/splits. The
source's 29 classes include J/Z (motion signs) and three non-letter labels;
the requested 24-output static alphabet model therefore uses A-I and K-Y.
No augmentation is performed here. Apply any training-only augmentation in a
future training Dataset, after loading the saved split manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError
from sklearn.model_selection import train_test_split

ImageFile.LOAD_TRUNCATED_IMAGES = False
SEED = 42
SIZE = (400, 400)
CLASSES = list("ABCDEFGHI") + list("KLMNOPQRSTUVWXY")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover_source(explicit: Path | None) -> Path:
    """Find the existing KaggleHub cache without calling dataset_download."""
    if explicit:
        return explicit.expanduser().resolve()
    cache = Path.home() / ".cache/kagglehub/datasets/grassknoted/asl-alphabet/versions"
    versions = sorted((p for p in cache.glob("*") if p.is_dir()), reverse=True) if cache.exists() else []
    for version in versions:
        candidates = list(version.glob("asl_alphabet_train/asl_alphabet_train"))
        if candidates:
            return candidates[0]
    raise FileNotFoundError("ASL dataset not found in KaggleHub cache; pass --source to its class-folder directory.")


def locate_class_root(source: Path) -> Path:
    """Accept either the class-folder directory or its parent download directory."""
    for candidate in (source, source / "asl_alphabet_train", source / "asl_alphabet_train" / "asl_alphabet_train"):
        if candidate.is_dir() and sum((candidate / c).is_dir() for c in CLASSES) >= 20:
            return candidate
    # Common extracted layout: .../asl_alphabet_train/asl_alphabet_train
    matches = [p for p in source.rglob("*") if p.is_dir() and sum((p / c).is_dir() for c in CLASSES) >= 20]
    if not matches:
        raise FileNotFoundError(f"Could not find alphabet class folders under {source}")
    return min(matches, key=lambda p: len(p.parts))


def source_images(root: Path) -> list[tuple[Path, str]]:
    """Enumerate only the chosen static alphabet classes in stable order."""
    return [(p, label) for label in CLASSES for p in sorted((root / label).rglob("*"))
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]


def image_digest(image: Image.Image) -> str:
    """Hash decoded source pixels, catching byte-different exact image duplicates."""
    rgb = image.convert("RGB")
    return hashlib.sha256(rgb.width.to_bytes(4, "big") + rgb.height.to_bytes(4, "big") + rgb.tobytes()).hexdigest()


def inspect_and_process(root: Path, output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read and audit every selected file, preserving originals and writing usable copies."""
    files = source_images(root)
    if not files:
        raise RuntimeError(f"No supported images found below {root}")
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    bad: list[str] = []
    dimensions: Counter[str] = Counter()
    formats: Counter[str] = Counter()
    hashes: dict[str, list[int]] = defaultdict(list)
    duplicates: list[dict[str, Any]] = []
    near_hashes: dict[tuple[str, str], list[int]] = defaultdict(list)
    for path, label in files:
        try:
            with Image.open(path) as im:
                im.load()
                dims, fmt = im.size, (im.format or path.suffix.lstrip(".")).upper()
                dimensions[f"{dims[0]}x{dims[1]}"] += 1
                formats[fmt] += 1
                digest = image_digest(im)
                if digest in hashes:
                    duplicates.append({"path": str(path), "duplicate_of": records[hashes[digest][0]]["source_path"], "sha256_pixels": digest})
                    continue
                # A compact perceptual signature identifies exact resized grayscale duplicates.
                gray = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                px = np.asarray(gray, dtype=np.uint8)
                bits = (px[:, 1:] > px[:, :-1]).flatten()
                d_hash = "".join("1" if b else "0" for b in bits)
                # Identical 64-bit dHashes are a conservative near-duplicate screen.
                # dHash is deliberately compared within the same class only: the
                # 64-bit signature is a coarse candidate screen and can collide
                # across different signs with similar backgrounds.
                near_hashes[(label, d_hash)].append(len(records))
                label_id = CLASSES.index(label)
                dst = output / label / f"{digest}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.exists():
                    im.convert("L").resize(SIZE, Image.Resampling.LANCZOS).save(dst, format="PNG", compress_level=1)
                idx = len(records)
                records.append({"source_path": str(path), "image_path": str(dst), "label": label,
                                "label_id": label_id, "width": dims[0], "height": dims[1],
                                "format": fmt, "sha256_pixels": digest, "dhash": d_hash})
                hashes[digest].append(idx)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            bad.append(f"{path}: {type(exc).__name__}: {exc}")
    # Same dHash values flag near duplicates for a conservative split grouping.
    for inds in near_hashes.values():
        if len(inds) > 1:
            for i in inds:
                records[i]["duplicate_group"] = min(inds)
    for idx, row in enumerate(records):
        row.setdefault("duplicate_group", idx)
    summary = {"source_root": str(root), "selected_source_images": len(files), "processed_unique_images": len(records),
               "dataset_structure": "KaggleHub extraction contains asl_alphabet_train/asl_alphabet_train/<class>/*.jpg (29 class folders) and asl_alphabet_test/asl_alphabet_test/*_test.jpg (28 images). This reproducible split uses the 24 selected static classes from the class-folder training directory; the small supplied test folder is retained untouched and not merged into the splits.",
               "classes": CLASSES, "class_counts": dict(Counter(r["label"] for r in records)),
               "excluded_source_classes": {"J": "dynamic/motion sign", "Z": "dynamic/motion sign", "del": "not an alphabet letter", "nothing": "not an alphabet letter", "space": "not an alphabet letter"},
               "source_dimensions": dict(dimensions), "source_formats": dict(formats),
               "corrupted_or_unreadable": bad, "exact_duplicates": duplicates,
               "near_duplicate_groups_same_class_dhash": sum(len(v) > 1 for v in near_hashes.values()),
               "near_duplicate_method": "Groups identical 64-bit difference hashes within a label; this conservative candidate screen does not detect all perceptual near-duplicates.",
               "processed_dimensions": [400, 400], "channels": 1,
               "normalization": "grayscale uint8 PNG; consumers convert to float32 and divide by 255 (range [0,1]); no dataset-fitted statistics",
               "temporal_content": "Static still images only; no video or sequence files were selected.",
               "group_limitation": "No signer, capture-session, or source-sequence metadata is provided in the class-folder dataset; image-level split cannot establish signer-independent generalization."}
    return records, summary


def make_splits(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Make reproducible stratified 70/15/15 splits while keeping duplicate groups intact."""
    df = pd.DataFrame(records)
    # Retain only uniquely labeled groups; conflicting duplicate labels are excluded from all splits.
    group_rows = []
    excluded: set[int] = set()
    for group, rows in df.groupby("duplicate_group"):
        labels = rows.label_id.unique()
        if len(labels) != 1:
            excluded.update(rows.index.tolist())
        else:
            group_rows.append((int(group), int(labels[0])))
    gdf = pd.DataFrame(group_rows, columns=["group", "label"])
    train_g, temp_g = train_test_split(gdf, test_size=0.30, random_state=SEED, stratify=gdf.label)
    val_g, test_g = train_test_split(temp_g, test_size=0.50, random_state=SEED, stratify=temp_g.label)
    result = {}
    for name, part in (("train", train_g), ("validation", val_g), ("test", test_g)):
        allowed = set(part.group)
        result[name] = [i for i, row in enumerate(records) if row["duplicate_group"] in allowed]
    result["excluded_conflicting_duplicates"] = sorted(excluded)
    return result


def write_report(records: list[dict[str, Any]], splits: dict[str, list[int]], summary: dict[str, Any], out: Path) -> None:
    """Write manifests and human-readable audit/split reports."""
    out.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    split_sets: dict[str, set[str]] = {}
    for name in ("train", "validation", "test"):
        idxs = splits[name]
        split_sets[name] = {records[i]["sha256_pixels"] for i in idxs}
        for i in idxs:
            row = records[i]
            manifest_rows.append({**row, "split": name})
    pd.DataFrame(manifest_rows).drop(columns=["dhash"]).to_csv(out / "manifest.csv", index=False)
    exact_cross = any(split_sets[a] & split_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    report = {**summary, "seed": SEED, "split_ratios": {"train": .70, "validation": .15, "test": .15},
              "split_counts": {}, "split_class_counts": {}, "exact_duplicates_cross_splits": exact_cross,
              "duplicate_groups_cross_splits": False,
              "excluded_conflicting_duplicate_count": len(splits["excluded_conflicting_duplicates"])}
    group_sets = {name: {records[i]["duplicate_group"] for i in splits[name]} for name in ("train", "validation", "test")}
    report["duplicate_groups_cross_splits"] = any(group_sets[a] & group_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")))
    for name in ("train", "validation", "test"):
        chosen = [records[i] for i in splits[name]]
        report["split_counts"][name] = len(chosen)
        report["split_class_counts"][name] = {c: sum(r["label"] == c for r in chosen) for c in CLASSES}
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"split_counts": report["split_counts"], "class_counts": report["class_counts"],
                      "split_class_counts": report["split_class_counts"], "source_dimensions": report["source_dimensions"],
                      "source_formats": report["source_formats"], "unreadable": len(report["corrupted_or_unreadable"]),
                      "processed": report["processed_unique_images"], "exact_duplicates": len(report["exact_duplicates"]),
                      "near_duplicate_groups": report["near_duplicate_groups_same_class_dhash"],
                      "exact_duplicates_cross_splits": report["exact_duplicates_cross_splits"],
                      "group_limitation": report["group_limitation"]}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Downloaded class-folder path; defaults to KaggleHub cache.")
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits"))
    args = parser.parse_args()
    root = locate_class_root(discover_source(args.source))
    print(f"Inspecting existing dataset at: {root}")
    records, summary = inspect_and_process(root, args.output)
    split_indices = make_splits(records)
    write_report(records, split_indices, summary, args.splits)


if __name__ == "__main__":
    main()
