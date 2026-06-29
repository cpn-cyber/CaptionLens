import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch
import yaml
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from torchvision import transforms
from tqdm import tqdm

from models.image_captioning import ImageCaptioningModel
from utils.decoding import decode_from_memory
from utils.tokenizer import Tokenizer


def load_references(captions_path, tokenizer):
    with open(captions_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    references = defaultdict(list)
    for annotation in data["annotations"]:
        caption = tokenizer.clean_caption(annotation["caption"])
        references[annotation["image_id"]].append(caption)
    return dict(references)


def load_image_ids(captions_path):
    with open(captions_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return {
        annotation["image_id"]
        for annotation in data["annotations"]
    }


def validate_evaluation_setup(config, tokenizer, metadata_path):
    train_path = config["dataset"]["train_captions_path"]
    validation_path = config["dataset"]["validation_captions_path"]
    train_ids = load_image_ids(train_path)
    validation_ids = load_image_ids(validation_path)
    overlap = train_ids & validation_ids
    if overlap:
        raise ValueError(
            f"Data leakage detected: {len(overlap)} image IDs occur in both "
            "the training and validation splits."
        )

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    expected_validation_path = os.path.normpath(validation_path)
    trained_validation_path = os.path.normpath(
        metadata["validation_captions_path"]
    )
    if trained_validation_path != expected_validation_path:
        raise ValueError(
            "The checkpoint metadata does not match the configured "
            "validation split. Retrain the model before evaluation."
        )

    if metadata["vocab_size"] != len(tokenizer.word2idx):
        raise ValueError(
            "The checkpoint vocabulary size does not match data/vocab.pkl."
        )


def compute_metrics(references, hypotheses):
    image_ids = sorted(hypotheses)
    ground_truth = {
        str(image_id): references[image_id]
        for image_id in image_ids
    }
    predictions = {
        str(image_id): [hypotheses[image_id]]
        for image_id in image_ids
    }

    bleu_scores, _ = Bleu(4).compute_score(
        ground_truth,
        predictions,
        verbose=0,
    )
    cider_score, _ = Cider().compute_score(ground_truth, predictions)

    return {
        "BLEU-1": float(bleu_scores[0]),
        "BLEU-2": float(bleu_scores[1]),
        "BLEU-3": float(bleu_scores[2]),
        "BLEU-4": float(bleu_scores[3]),
        "CIDEr": float(cider_score),
    }


def load_model(config, tokenizer, checkpoint_path, device):
    model = ImageCaptioningModel(
        vocab_size=len(tokenizer.word2idx),
        embed_size=config["model"]["embed_size"],
        decoder_dim=config["model"]["decoder_dim"],
        attention_dim=config["model"]["attention_dim"],
        dropout=config["model"]["dropout"],
        max_len=config["model"]["max_len"],
        use_semantic_slots=config["model"].get("use_semantic_slots", False),
        num_semantic_slots=config["model"].get("num_semantic_slots", 16),
        semantic_slot_layers=config["model"].get("semantic_slot_layers", 2),
        semantic_slot_heads=config["model"].get("semantic_slot_heads", 8),
        include_visual_tokens=config["model"].get("include_visual_tokens", True),
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def build_comparison(results):
    if "greedy" not in results or "beam" not in results:
        return {}

    comparison = {}
    for metric, greedy_score in results["greedy"].items():
        beam_score = results["beam"][metric]
        absolute_change = beam_score - greedy_score
        relative_change = (
            absolute_change / greedy_score * 100
            if greedy_score != 0
            else None
        )
        comparison[metric] = {
            "absolute_change": absolute_change,
            "relative_change_percent": relative_change,
            "improved": absolute_change > 0,
        }
    return comparison


def print_results(results, comparison):
    headers = ["Strategy", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "CIDEr"]
    print("\n" + " | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for strategy, scores in results.items():
        values = [strategy]
        values.extend(f"{scores[metric]:.4f}" for metric in headers[1:])
        print(" | ".join(values))

    if comparison:
        print("\nBeam Search change relative to Greedy:")
        for metric, values in comparison.items():
            relative = values["relative_change_percent"]
            relative_text = "N/A" if relative is None else f"{relative:+.2f}%"
            print(
                f"- {metric}: {values['absolute_change']:+.4f} "
                f"({relative_text})"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CaptionLens with BLEU-1..4 and CIDEr."
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["greedy", "beam"],
        default=["greedy", "beam"],
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Evaluate only the first N validation images for a quick check.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    captions_path = config["dataset"]["validation_captions_path"]
    image_dir = config["dataset"].get(
        "validation_image_dir",
        config["dataset"]["image_dir"],
    )
    vocab_path = config["processed_data"]["vocab_path"]
    checkpoint_path = os.path.join(
        config["train"]["save_dir"],
        "best_model.pth",
    )
    metadata_path = os.path.join(
        config["train"]["save_dir"],
        "training_metadata.json",
    )
    output_path = args.output or config["evaluation"]["output_path"]

    required_paths = [
        captions_path,
        config["dataset"]["train_captions_path"],
        image_dir,
        vocab_path,
        checkpoint_path,
        metadata_path,
    ]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Evaluation resources are missing:\n"
            f"{missing}\n"
            "Run prepare_splits.py, build_vocab.py, and train.py first."
        )

    tokenizer = Tokenizer()
    tokenizer.load_vocab(vocab_path)
    validate_evaluation_setup(config, tokenizer, metadata_path)
    references = load_references(captions_path, tokenizer)
    image_ids = sorted(references)
    if args.max_images is not None:
        image_ids = image_ids[:args.max_images]

    model = load_model(config, tokenizer, checkpoint_path, device)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    decoding_config = dict(config["decoding"])
    hypotheses = {strategy: {} for strategy in args.strategies}

    for image_id in tqdm(image_ids, desc=f"Evaluating on {device.type}"):
        image_path = os.path.join(image_dir, f"{image_id:012d}.jpg")
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            if hasattr(model, "encode_image"):
                memory = model.encode_image(image_tensor)
            else:
                memory = model.encoder(image_tensor)
            for strategy in args.strategies:
                caption = decode_from_memory(
                    model,
                    memory,
                    tokenizer,
                    strategy=strategy,
                    **decoding_config,
                )
                hypotheses[strategy][image_id] = tokenizer.clean_caption(caption)

    results = {
        strategy: compute_metrics(references, strategy_hypotheses)
        for strategy, strategy_hypotheses in hypotheses.items()
    }
    comparison = build_comparison(results)

    report = {
        "dataset": captions_path,
        "checkpoint": checkpoint_path,
        "device": device.type,
        "evaluated_images": len(image_ids),
        "decoding": decoding_config,
        "metrics": results,
        "beam_vs_greedy": comparison,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print_results(results, comparison)
    print(f"\nSaved evaluation report to: {output_path}")


if __name__ == "__main__":
    main()
