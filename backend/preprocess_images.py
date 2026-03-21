from pathlib import Path

DATA_DIR = Path("../data/raw/images/train")

def count_images():
    total = 0
    for folder in DATA_DIR.iterdir():
        if folder.is_dir():
            images = list(folder.glob("*"))
            print(f"{folder.name}: {len(images)} images")
            total += len(images)
    print(f"\nTOTAL: {total}")

if __name__ == "__main__":
    count_images()