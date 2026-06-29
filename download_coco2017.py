import argparse
import os
import urllib.request
import zipfile
from pathlib import Path


URLS = {
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
}


def download(url, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Found existing file: {destination}")
        return

    print(f"Downloading {url}")
    print(f"Saving to {destination}")
    urllib.request.urlretrieve(url, destination)


def extract(zip_path, destination):
    zip_path = Path(zip_path)
    destination = Path(destination)
    marker = destination / f".{zip_path.stem}_extracted"
    if marker.exists():
        print(f"Already extracted: {zip_path}")
        return

    print(f"Extracting {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(destination)
    marker.touch()


def count_images(path):
    path = Path(path)
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*.jpg") if item.is_file())


def main():
    parser = argparse.ArgumentParser(
        description="Download the official MSCOCO 2017 train/val captioning data."
    )
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Download annotations only.",
    )
    parser.add_argument(
        "--delete-zip",
        action="store_true",
        help="Delete ZIP files after successful extraction.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    annotation_zip = data_root / "annotations_trainval2017.zip"
    download(URLS["annotations"], annotation_zip)
    extract(annotation_zip, data_root)

    if not args.skip_images:
        for split in ["train2017", "val2017"]:
            zip_path = data_root / f"{split}.zip"
            download(URLS[split], zip_path)
            extract(zip_path, data_root)

    train_count = count_images(data_root / "train2017")
    val_count = count_images(data_root / "val2017")
    print(f"train2017 images: {train_count}")
    print(f"val2017 images: {val_count}")
    print(f"annotations: {data_root / 'annotations'}")

    if args.delete_zip:
        for name in ["annotations_trainval2017", "train2017", "val2017"]:
            zip_path = data_root / f"{name}.zip"
            if zip_path.exists():
                os.remove(zip_path)
                print(f"Deleted {zip_path}")


if __name__ == "__main__":
    main()
