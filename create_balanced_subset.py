"""Create a capped, class-balanced copy of the image training dataset."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


SOURCE_DIR = Path("data/raw/images/train")
OUTPUT_DIR = Path("data/raw/images/train_balanced")
MAX_IMAGES_PER_CLASS = 150
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def destination_for(source: Path, class_dir: Path) -> Path:
    """Return a stable, unique output path for a source image."""
    relative_path = source.relative_to(SOURCE_DIR).as_posix()
    path_hash = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
    return OUTPUT_DIR / class_dir.name / f"{source.stem}__{path_hash}{source.suffix.lower()}"


def main() -> None:
    if not SOURCE_DIR.is_dir():
        print(f"Source folder not found: {SOURCE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for class_dir in sorted(path for path in SOURCE_DIR.iterdir() if path.is_dir()):
        images = sorted(path for path in class_dir.rglob("*") if is_image(path))
        selected_images = images[:MAX_IMAGES_PER_CLASS]
        output_class_dir = OUTPUT_DIR / class_dir.name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for source in selected_images:
            destination = destination_for(source, class_dir)
            if not destination.exists():
                shutil.copy2(source, destination)
                copied += 1

        final_count = sum(1 for path in output_class_dir.rglob("*") if is_image(path))
        print(f"{class_dir.name}: {final_count} image(s) ({copied} copied this run)")


if __name__ == "__main__":
    main()
