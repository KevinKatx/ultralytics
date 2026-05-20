"""
YOLO Dataset Balancer
=====================
Undersamples the dominant class and oversamples underrepresented classes
to match target distribution for YOLOv8 / YOLO11 training.

Expected Roboflow export structure:
    dataset/
        train/
            images/   *.jpg / *.png
            labels/   *.txt  (YOLO format)
        valid/
            images/
            labels/
        test/
            images/
            labels/

Usage:
    python balance_dataset.py --dataset ./dataset --output ./dataset_balanced
"""

import os
import random
import shutil
import argparse
import sys
from collections import defaultdict
from pathlib import Path

# ── Target instance counts per class (edit as needed) ──────────────────────
CLASS_TARGETS = {
    "ManMade_objects":          500,
    "plastic_debris":           500,
    "Submerged_metal":          500,
    "Submerged_natural_object": 500,
    "Submerged_wood":           500,
}

# ── Class index → name mapping (must match your data.yaml order) ────────────
CLASS_NAMES = {
    0: "ManMade_objects",
    1: "Submerged_metal",
    2: "Submerged_natural_object",
    3: "Submerged_wood",
    4: "plastic_debris",
}


SPLITS = ["train", "valid", "test"]

# ── Augmentation settings ───────────────────────────────────────────────────
# Only train split is augmented; valid/test are just copied as-is.
AUGMENT_SPLITS = {"train"}

# How many augmented copies to try generating per original image
MAX_AUG_COPIES = 5

# Random seed for reproducibility
SEED = 42


def parse_label(path: Path):
    """Return list of (class_id, cx, cy, w, h) from a YOLO .txt file."""
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 5:
                rows.append((int(p[0]), float(p[1]), float(p[2]),
                              float(p[3]), float(p[4])))
    return rows
 
 
def write_label(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(f"{r[0]} {r[1]:.6f} {r[2]:.6f} {r[3]:.6f} {r[4]:.6f}\n")
 
 
def find_image(images_dir: Path, stem: str):
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        p = images_dir / (stem + ext)
        if p.exists():
            return p
    return None
 
 
def copy_pair(img_src: Path, lbl_src: Path,
              img_dst_dir: Path, lbl_dst_dir: Path,
              rows=None):
    """Copy image and label (optionally with filtered rows) to destination."""
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_src, img_dst_dir / img_src.name)
    if rows is None:
        shutil.copy2(lbl_src, lbl_dst_dir / (lbl_src.stem + ".txt"))
    else:
        write_label(lbl_dst_dir / (lbl_src.stem + ".txt"), rows)
 
 
def augment_image(img_path: Path, seed_val: int):
    """
    Returns (bytes_or_None, was_flipped).
    Applies brightness/saturation/contrast jitter + optional H-flip via Pillow.
    Falls back to (None, False) if Pillow is unavailable.
    """
    try:
        from PIL import Image, ImageEnhance, ImageOps
        import io
        img = Image.open(img_path).convert("RGB")
        r = random.Random(seed_val)
        flipped = r.random() < 0.5
        if flipped:
            img = ImageOps.mirror(img)
        img = ImageEnhance.Brightness(img).enhance(r.uniform(0.80, 1.20))
        img = ImageEnhance.Color(img).enhance(r.uniform(0.75, 1.25))
        img = ImageEnhance.Contrast(img).enhance(r.uniform(0.85, 1.15))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return buf.getvalue(), flipped
    except ImportError:
        return None, False
 
 
def write_aug_pair(img_src: Path, rows, img_dst_dir: Path, lbl_dst_dir: Path,
                   new_stem: str, seed_val: int):
    """Write one augmented image+label pair."""
    img_bytes, flipped = augment_image(img_src, seed_val)
    aug_rows = rows
    if flipped:
        aug_rows = [(c, 1.0 - cx, cy, w, h) for c, cx, cy, w, h in rows]
    img_dst_dir.mkdir(parents=True, exist_ok=True)
    lbl_dst_dir.mkdir(parents=True, exist_ok=True)
    dst = img_dst_dir / (new_stem + ".jpg")
    if img_bytes:
        dst.write_bytes(img_bytes)
    else:
        shutil.copy2(img_src, dst)
    write_label(lbl_dst_dir / (new_stem + ".txt"), aug_rows)
 
 
def count_from_disk(lbl_dir: Path):
    counts = defaultdict(int)
    for f in lbl_dir.glob("*.txt"):
        for row in parse_label(f):
            counts[row[0]] += 1
    return counts
 
 
# ───────────────────────────────────────────────────────────────────────────
# Collect split
# ───────────────────────────────────────────────────────────────────────────
 
def collect_split(split_dir: Path):
    """
    Returns:
        per_class      : {class_id: [(img_path, lbl_path), ...]}
        all_samples    : [(img_path, lbl_path)]
        lbl_to_classes : {lbl_path: set_of_class_ids}
    """
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        return defaultdict(list), [], {}
 
    per_class      = defaultdict(list)
    all_samples    = []
    lbl_to_classes = {}
 
    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        img_file = find_image(img_dir, lbl_file.stem)
        if img_file is None:
            continue
        rows = parse_label(lbl_file)
        if not rows:
            continue
        cids = {r[0] for r in rows}
        lbl_to_classes[lbl_file] = cids
        all_samples.append((img_file, lbl_file))
        for cid in cids:
            per_class[cid].append((img_file, lbl_file))
 
    return per_class, all_samples, lbl_to_classes
 
 
# ───────────────────────────────────────────────────────────────────────────
# Main balancing logic
# ───────────────────────────────────────────────────────────────────────────
 
def balance_split(split_dir: Path, out_split_dir: Path,
                  do_augment: bool, rng: random.Random):
 
    print(f"\n{'='*60}")
    print(f"  Processing split: {split_dir.name}")
    print(f"{'='*60}")
 
    per_class, all_samples, lbl_to_classes = collect_split(split_dir)
    if not all_samples:
        print("  [skip] No labelled samples found.")
        return
 
    # Instance counts before any changes
    before_counts = defaultdict(int)
    for _, lbl in all_samples:
        for r in parse_label(lbl):
            before_counts[r[0]] += 1
 
    print("\n  Before balancing (instances per class):")
    for cid, name in CLASS_NAMES.items():
        print(f"    {name:<35} {before_counts.get(cid, 0):>6}")
 
    out_img_dir = out_split_dir / "images"
    out_lbl_dir = out_split_dir / "labels"
 
    # Identify dominant vs minority class ids for this split
    dominant_cids = {
        cid for cid, name in CLASS_NAMES.items()
        if CLASS_TARGETS.get(name, 0) < before_counts.get(cid, 0)
    }
    minority_cids_set = set(CLASS_NAMES.keys()) - dominant_cids
 
    # ── Step 1: Undersampling ─────────────────────────────────────────────
    # discard   → drop image entirely (no minority annotations in it)
    # strip_set → keep image but remove dominant-class rows from label
 
    discard:   set[Path] = set()
    strip_set: set[Path] = set()
 
    for cid in dominant_cids:
        name = CLASS_NAMES[cid]
        imgs_for_class = list(per_class.get(cid, []))
        total = len(imgs_for_class)
 
        current = before_counts.get(cid, 0)
        avg     = current / max(total, 1)
        target  = CLASS_TARGETS.get(name, current)
        needed  = max(1, round(target / avg))
 
        if needed >= total:
            continue
 
        rng.shuffle(imgs_for_class)
        # Mark excess images (beyond what we need to keep)
        excess_lbls = {lbl for _, lbl in imgs_for_class[needed:]}
 
        n_discarded = 0
        n_stripped  = 0
        for lbl in excess_lbls:
            if lbl in discard:
                continue
            has_minority = bool(lbl_to_classes[lbl] & minority_cids_set)
            if has_minority:
                # Must keep image for minority data — strip dominant rows instead
                strip_set.add(lbl)
                n_stripped += 1
            else:
                # Pure dominant image — safe to drop entirely
                discard.add(lbl)
                n_discarded += 1
 
        print(f"\n  [Undersample] {name}: "
              f"target ~{needed} imgs | "
              f"discarded {n_discarded} pure | "
              f"stripped annotations in {n_stripped} mixed")
 
    # ── Step 2: Write originals to output ─────────────────────────────────
    pool = [(img, lbl) for img, lbl in all_samples if lbl not in discard]
 
    for img_src, lbl_src in pool:
        if lbl_src in strip_set:
            # Remove dominant-class rows from the copied label
            rows    = parse_label(lbl_src)
            cleaned = [r for r in rows if r[0] not in dominant_cids]
            if not cleaned:
                continue  # would produce empty label — skip image
            copy_pair(img_src, lbl_src, out_img_dir, out_lbl_dir, rows=cleaned)
        else:
            copy_pair(img_src, lbl_src, out_img_dir, out_lbl_dir)
 
    # ── Step 3: Oversampling (train split only) ───────────────────────────
    if do_augment:
        actual = count_from_disk(out_lbl_dir)
 
        for cid in sorted(minority_cids_set):
            name    = CLASS_NAMES[cid]
            target  = CLASS_TARGETS.get(name, 0)
            current = actual.get(cid, 0)
            if current >= target:
                continue
 
            needed_extra = target - current
 
            # Prefer images that contain ONLY this minority class
            # so we never copy dominant-class annotations into the output
            pure_sources = [
                (img, lbl) for img, lbl in per_class.get(cid, [])
                if not (lbl_to_classes.get(lbl, set()) & dominant_cids)
            ]
            mixed_sources = [
                (img, lbl) for img, lbl in per_class.get(cid, [])
                if lbl_to_classes.get(lbl, set()) & dominant_cids
            ]
            # Use pure pool; only fall back to mixed if pool is too small
            source_imgs = pure_sources if len(pure_sources) >= 3 else (pure_sources + mixed_sources)
 
            if not source_imgs:
                print(f"\n  [Skip oversample] {name}: no source images available.")
                continue
 
            # Count this class's instances in chosen source pool
            pool_instances = sum(
                sum(1 for r in parse_label(lbl) if r[0] == cid)
                for _, lbl in source_imgs
            )
            avg_per_img = pool_instances / max(len(source_imgs), 1)
            needed_imgs = max(1, round(needed_extra / max(avg_per_img, 1)))
 
            print(f"\n  [Oversample] {name}: "
                  f"current={current} target={target} "
                  f"need +{needed_extra} instances (~{needed_imgs} aug imgs) "
                  f"| pure_pool={len(pure_sources)} mixed_pool={len(mixed_sources)}")
 
            aug_count = 0
            for pass_i in range(MAX_AUG_COPIES):
                if aug_count >= needed_imgs:
                    break
                cycle = list(source_imgs)
                rng.shuffle(cycle)
                for img_src, lbl_src in cycle:
                    if aug_count >= needed_imgs:
                        break
                    # Write ONLY this class's rows — never carry dominant annotations
                    all_rows    = parse_label(lbl_src)
                    target_rows = [r for r in all_rows if r[0] == cid]
                    if not target_rows:
                        continue
                    new_stem = f"{img_src.stem}_aug{pass_i}_{aug_count}"
                    seed_val = pass_i * 100000 + aug_count
                    write_aug_pair(img_src, target_rows,
                                   out_img_dir, out_lbl_dir,
                                   new_stem, seed_val)
                    aug_count += 1
 
            print(f"    → Generated {aug_count} augmented images for {name}")
 
    # ── Final report ─────────────────────────────────────────────────────
    final      = count_from_disk(out_lbl_dir)
    total_imgs = len(list(out_img_dir.glob("*")))
    print(f"\n  After balancing ({total_imgs} total images):")
    for cid, name in CLASS_NAMES.items():
        b    = before_counts.get(cid, 0)
        a    = final.get(cid, 0)
        t    = CLASS_TARGETS.get(name, "—")
        d    = a - b
        sign = "+" if d >= 0 else ""
        print(f"    {name:<35} {b:>6} → {a:>6}  (target: {t}, {sign}{d})")
 
 
# ───────────────────────────────────────────────────────────────────────────
# data.yaml helper
# ───────────────────────────────────────────────────────────────────────────
 
def copy_data_yaml(src_dir: Path, dst_dir: Path):
    for fname in ("data.yaml", "dataset.yaml"):
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)
            print(f"\n  Copied {fname} to output directory.")
            return
    yaml_path = dst_dir / "data.yaml"
    lines = [
        "path: .\n",
        "train: train/images\n",
        "val:   valid/images\n",
        "test:  test/images\n\n",
        f"nc: {len(CLASS_NAMES)}\n",
        "names:\n",
    ]
    for cid in sorted(CLASS_NAMES):
        lines.append(f"  {cid}: {CLASS_NAMES[cid]}\n")
    yaml_path.write_text("".join(lines))
    print(f"\n  Created minimal data.yaml at {yaml_path}")
 
 
# ───────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Balance a Roboflow YOLO dataset by undersampling "
                    "dominant classes and oversampling minority classes."
    )
    parser.add_argument("--dataset", required=True,
                        help="Path to the Roboflow dataset root "
                             "(contains train/, valid/, test/).")
    parser.add_argument("--output",  required=True,
                        help="Output path for the balanced dataset.")
    parser.add_argument("--seed",    type=int, default=SEED,
                        help=f"Random seed (default: {SEED}).")
    parser.add_argument("--no-augment", action="store_true",
                        help="Skip oversampling; only undersample.")
    args = parser.parse_args()
 
    dataset_dir = Path(args.dataset).resolve()
    output_dir  = Path(args.output).resolve()
 
    if not dataset_dir.exists():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)
 
    if output_dir.exists():
        ans = input(f"Output directory '{output_dir}' already exists. Overwrite? [y/N] ")
        if ans.strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)
        shutil.rmtree(output_dir)
 
    output_dir.mkdir(parents=True)
    rng = random.Random(args.seed)
 
    print(f"\nDataset : {dataset_dir}")
    print(f"Output  : {output_dir}")
    print(f"Seed    : {args.seed}")
    print(f"Augment : {not args.no_augment}")
    print(f"\nTargets:")
    for name, count in CLASS_TARGETS.items():
        print(f"  {name:<35} {count:>6}")
 
    for split in SPLITS:
        split_dir = dataset_dir / split
        if not split_dir.exists():
            print(f"\n  [skip] Split '{split}' not found.")
            continue
        do_augment = (split in AUGMENT_SPLITS) and (not args.no_augment)
        balance_split(split_dir, output_dir / split, do_augment, rng)
 
    copy_data_yaml(dataset_dir, output_dir)
 
    print("\n" + "="*60)
    print("  Done! Balanced dataset saved to:")
    print(f"  {output_dir}")
    print("="*60)
 
 
if __name__ == "__main__":
    main()