"""Build a normalized training image dataset from the Kaggle staging folders."""

from __future__ import annotations

import hashlib
import shutil
from collections import Counter
from pathlib import Path


SOURCE_DIR = Path("data/raw/images/kaggle_tmp")
OUTPUT_DIR = Path("data/raw/images/train")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def contains(path: str, text: str) -> bool:
    """Perform a case-insensitive match against a source folder path."""
    return text.casefold() in path.casefold()


def detect_class(parent_path: Path) -> str | None:
    """Return the mapped class for a source image's parent path, if supported.

    Rules are intentionally ordered from most specific to most general.
    """
    path = parent_path.as_posix()

    if contains(path, "Tinea Ringworm Candidiasis"):
        return "Fungal Infection"
    if contains(path, "Atopic Dermatitis"):
        return "Atopic Dermatitis"
    if contains(path, "Acne and Rosacea"):
        return "Acne and Rosacea"
    if contains(path, "BA- cellulitis") or contains(path, "BA-cellulitis"):
        return "Cellulitis"
    if contains(path, "BA-impetigo"):
        return "Impetigo"
    if contains(path, "FU-athlete-foot"):
        return "Athlete Foot"
    if contains(path, "FU-nail-fungus"):
        return "Nail Fungus"
    if contains(path, "FU-ringworm"):
        return "Ringworm"
    if contains(path, "VI-chickenpox"):
        return "Chickenpox"
    if contains(path, "VI-shingles"):
        return "Shingles"
    if contains(path, "Basal Cell Carcinoma") or contains(path, "BCC"):
        return "Basal Cell Carcinoma"
    if contains(path, "Actinic keratosis"):
        return "Actinic Keratosis"
    if contains(path, "Benign keratosis"):
        return "Benign Keratosis"
    if contains(path, "Melanocytic") or contains(path, "Nevi"):
        return "Melanocytic Nevus"
    if contains(path, "Warts Molluscum"):
        return "Warts and Molluscum"
    if contains(path, "Vascular lesion") or contains(path, "Vascular Tumors"):
        return "Vascular Lesion"
    if contains(path, "Cellulitis") and not contains(path, "Impetigo"):
        return "Cellulitis"
    if contains(path, "Impetigo") and not contains(path, "Cellulitis"):
        return "Impetigo"
    if contains(path, "Eczema"):
        return "Eczema"
    if contains(path, "Psoriasis"):
        return "Psoriasis"
    if contains(path, "Melanoma"):
        return "Melanoma"

    return None


def destination_for(source: Path, disease_class: str) -> Path:
    """Create a stable destination name that cannot collide across source paths."""
    relative_path = source.relative_to(SOURCE_DIR).as_posix()
    suffix = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
    return OUTPUT_DIR / disease_class / f"{source.stem}__{suffix}{source.suffix.lower()}"


def image_count(directory: Path) -> int:
    return sum(
        1
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def main() -> None:
    if not SOURCE_DIR.is_dir():
        print(f"Source folder not found: {SOURCE_DIR}")
        return

    copied = 0
    skipped = 0
    classes_seen: set[str] = set()

    for source in SOURCE_DIR.rglob("*"):
        if not source.is_file() or source.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        disease_class = detect_class(source.parent)
        if disease_class is None:
            skipped += 1
            continue

        classes_seen.add(disease_class)
        destination = destination_for(source, disease_class)
        destination.parent.mkdir(parents=True, exist_ok=True)

        # The path-derived hash makes reruns safe and prevents overwrites.
        if not destination.exists():
            shutil.copy2(source, destination)
            copied += 1

    counts = Counter(
        {
            disease_class: image_count(OUTPUT_DIR / disease_class)
            for disease_class in classes_seen
            if (OUTPUT_DIR / disease_class).is_dir()
        }
    )

    print(f"Copied {copied} image(s); ignored {skipped} unmatched image(s).")
    print("Final image count per class:")
    for disease_class in sorted(counts):
        print(f"- {disease_class}: {counts[disease_class]}")


if __name__ == "__main__":
    main()
