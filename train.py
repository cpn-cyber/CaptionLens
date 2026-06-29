import json
import os

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from models.image_captioning import ImageCaptioningModel
from utils.dataset import MSCOCODataset
from utils.tokenizer import Tokenizer


def calculate_loss(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, captions in dataloader:
            images = images.to(device)
            captions = captions.to(device)
            outputs = model(images, captions[:, :-1])
            targets = captions[:, 1:]
            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                targets.reshape(-1),
            )
            total_loss += loss.item()

    return total_loss / len(dataloader)


def save_training_metadata(config, tokenizer, checkpoint_dir):
    metadata = {
        "train_captions_path": config["dataset"]["train_captions_path"],
        "validation_captions_path": config["dataset"]["validation_captions_path"],
        "validation_ratio": config["dataset"]["validation_ratio"],
        "split_seed": config["dataset"]["split_seed"],
        "vocab_path": config["processed_data"]["vocab_path"],
        "vocab_size": len(tokenizer.word2idx),
        "model": {
            "embed_size": config["model"]["embed_size"],
            "decoder_dim": config["model"]["decoder_dim"],
            "use_semantic_slots": config["model"].get("use_semantic_slots", False),
            "num_semantic_slots": config["model"].get("num_semantic_slots", 0),
            "semantic_slot_layers": config["model"].get("semantic_slot_layers", 0),
            "semantic_slot_heads": config["model"].get("semantic_slot_heads", 0),
            "include_visual_tokens": config["model"].get("include_visual_tokens", True),
        },
    }
    metadata_path = os.path.join(checkpoint_dir, "training_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)


def main():
    with open("config/config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    train_captions_path = config["dataset"]["train_captions_path"]
    validation_captions_path = config["dataset"]["validation_captions_path"]
    required_paths = [
        train_captions_path,
        validation_captions_path,
        config["processed_data"]["vocab_path"],
    ]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        missing = "\n".join(f"- {path}" for path in missing_paths)
        raise FileNotFoundError(
            "Training resources are missing:\n"
            f"{missing}\n"
            "Run prepare_splits.py and build_vocab.py first."
        )

    tokenizer = Tokenizer()
    tokenizer.load_vocab(config["processed_data"]["vocab_path"])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    dataset_kwargs = {
        "image_dir": config["dataset"]["image_dir"],
        "tokenizer": tokenizer,
        "transform": transform,
        "max_len": config["model"]["max_len"],
    }
    train_dataset = MSCOCODataset(
        image_dir=config["dataset"].get(
            "train_image_dir",
            config["dataset"]["image_dir"],
        ),
        captions_path=train_captions_path,
        tokenizer=tokenizer,
        transform=transform,
        max_len=config["model"]["max_len"],
    )
    validation_dataset = MSCOCODataset(
        image_dir=config["dataset"].get(
            "validation_image_dir",
            config["dataset"]["image_dir"],
        ),
        captions_path=validation_captions_path,
        tokenizer=tokenizer,
        transform=transform,
        max_len=config["model"]["max_len"],
    )

    num_workers = int(config["train"].get("num_workers", 0))
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

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
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.word2idx["<PAD>"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )

    best_validation_loss = float("inf")
    for epoch in range(config["train"]["num_epochs"]):
        model.train()
        total_train_loss = 0.0

        for step, (images, captions) in enumerate(train_loader, start=1):
            images = images.to(device)
            captions = captions.to(device)
            optimizer.zero_grad()

            outputs = model(images, captions[:, :-1])
            targets = captions[:, 1:]
            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                targets.reshape(-1),
            )
            loss.backward()

            clip_grad = config["train"].get("clip_grad")
            if clip_grad:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    clip_grad,
                )

            optimizer.step()
            total_train_loss += loss.item()

            if step % config["train"]["log_interval"] == 0:
                print(
                    f"Epoch [{epoch + 1}/{config['train']['num_epochs']}], "
                    f"Step [{step}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss = total_train_loss / len(train_loader)
        validation_loss = calculate_loss(
            model,
            validation_loader,
            criterion,
            device,
        )
        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.4f}, "
            f"validation_loss={validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            checkpoint_dir = config["train"]["save_dir"]
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir,
                "best_model.pth",
            )
            torch.save(model.state_dict(), checkpoint_path)
            save_training_metadata(config, tokenizer, checkpoint_dir)
            print(f"Best model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
