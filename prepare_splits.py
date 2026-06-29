import argparse
import json
import random
from pathlib import Path

import yaml


def split_annotations(data, validation_ratio, seed):
    annotations = data["annotations"]
    image_ids = sorted({annotation["image_id"] for annotation in annotations})

    random.Random(seed).shuffle(image_ids)
    validation_count = max(1, round(len(image_ids) * validation_ratio))
    validation_ids = set(image_ids[:validation_count])
    training_ids = set(image_ids[validation_count:])

    def build_split(selected_ids):
        split = {
            key: value
            for key, value in data.items()
            if key not in {"annotations", "images"}
        }
        split["annotations"] = [
            annotation
            for annotation in annotations
            if annotation["image_id"] in selected_ids
        ]
        if "images" in data:
            split["images"] = [
                image
                for image in data["images"]
                if image["id"] in selected_ids
            ]
        return split

    return (
        build_split(training_ids),
        build_split(validation_ids),
        training_ids,
        validation_ids,
    )


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Create deterministic image-level train/validation splits."
    )
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    dataset_config = config["dataset"]
    source_path = dataset_config["captions_path"]
    train_path = dataset_config["train_captions_path"]
    validation_path = dataset_config["validation_captions_path"]
    validation_ratio = float(dataset_config.get("validation_ratio", 0.1))
    seed = int(dataset_config.get("split_seed", 42))

    if not 0 < validation_ratio < 1:
        raise ValueError("dataset.validation_ratio must be between 0 and 1.")

    if Path(train_path).resolve() == Path(source_path).resolve():
        raise ValueError(
            "train_captions_path points to captions_path. Refusing to "
            "overwrite the source annotations. Pass separate split output "
            "paths or use the official COCO train/val files directly."
        )

    with open(source_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    train_data, validation_data, train_ids, validation_ids = split_annotations(
        data,
        validation_ratio,
        seed,
    )
    write_json(train_path, train_data)
    write_json(validation_path, validation_data)

    overlap = train_ids & validation_ids
    if overlap:
        raise RuntimeError("Train and validation image IDs overlap.")

    print(f"Train images: {len(train_ids)}")
    print(f"Validation images: {len(validation_ids)}")
    print(f"Train annotations: {len(train_data['annotations'])}")
    print(f"Validation annotations: {len(validation_data['annotations'])}")
    print(f"Saved train split to: {train_path}")
    print(f"Saved validation split to: {validation_path}")


if __name__ == "__main__":
    main()
