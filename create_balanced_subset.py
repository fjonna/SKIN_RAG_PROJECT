"""Create a capped, class-balanced copy of the image training dataset.

Two things drive the settings here, both measured in evaluation/:

1. The cap is 1000, not 150. Cross-validated over 400 held-out cases, growing
   the index from 2,860 to 12,097 images lifts image-only Top-1 from 59.8% to
   71.0% and fused Top-1 from 79.5% to 87.8%. The cap still matters -- without
   one, Melanocytic Nevus alone would contribute 8,099 images against Actinic
   Keratosis's 100 -- but 150 was leaving most of the corpus unused.

2. Images listed in the hold-out manifest are never copied. The evaluation set
   is reserved BEFORE the index is built, so no test image can reach the index
   at any cap. Reserving first also means classes with few images (Actinic
   Keratosis, Cutaneous Larva Migrans) still have test cases, which the old
   leftover-based split could not give them.

Regenerate the manifest with evaluation/index_size_experiment.py if the split
ever needs to change; everything downstream reads it from disk rather than
recomputing it, so the two cannot drift apart.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path


SOURCE_DIR = Path("data/raw/images/train")
OUTPUT_DIR = Path("data/raw/images/train_balanced")
HELDOUT_MANIFEST = Path("evaluation/results/heldout_test_manifest.csv")
MAX_IMAGES_PER_CLASS = 1000
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def load_heldout() -> set[str]:
    """Repo-relative POSIX paths of images reserved for evaluation."""
    if not HELDOUT_MANIFEST.exists():
        print(f"WARNING: no hold-out manifest at {HELDOUT_MANIFEST}; "
              f"nothing will be excluded and evaluation results will not be valid.")
        return set()
    with open(HELDOUT_MANIFEST, encoding="utf-8") as f:
        return {row["image_path"] for row in csv.DictReader(f)}


def destination_for(source: Path, class_dir: Path) -> Path:
    """Return a stable, unique output path for a source image."""
    relative_path = source.relative_to(SOURCE_DIR).as_posix()
    path_hash = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
    return OUTPUT_DIR / class_dir.name / f"{source.stem}__{path_hash}{source.suffix.lower()}"


def main() -> None:
    if not SOURCE_DIR.is_dir():
        print(f"Source folder not found: {SOURCE_DIR}")
        return

    heldout = load_heldout()
    print(f"{len(heldout)} image(s) reserved for evaluation will be skipped.\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0

    for class_dir in sorted(path for path in SOURCE_DIR.iterdir() if path.is_dir()):
        images = [p for p in sorted(class_dir.rglob("*")) if is_image(p)]
        eligible = [p for p in images if p.as_posix() not in heldout]
        selected_images = eligible[:MAX_IMAGES_PER_CLASS]
        output_class_dir = OUTPUT_DIR / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        # Drop anything a previous, differently-capped run left behind, so the
        # folder always reflects exactly the current selection.
        wanted = {destination_for(source, class_dir) for source in selected_images}
        for stale in list(output_class_dir.rglob("*")):
            if is_image(stale) and stale not in wanted:
                stale.unlink()

        copied = 0
        for source in selected_images:
            destination = destination_for(source, class_dir)
            if not destination.exists():
                shutil.copy2(source, destination)
                copied += 1

        final_count = sum(1 for path in output_class_dir.rglob("*") if is_image(path))
        total += final_count
        print(f"{class_dir.name:26s} {final_count:>5d} image(s)  "
              f"({copied} copied, {len(images) - len(eligible)} held out)")

    print(f"\nTotal indexed images: {total}")


if __name__ == "__main__":
    main()
