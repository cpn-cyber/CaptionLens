# train.py

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.tokenizer import Tokenizer
from utils.dataset import MSCOCODataset
from models.image_captioning import ImageCaptioningModel

def main():
    # Load config
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = Tokenizer()
    tokenizer.load_vocab(config['processed_data']['vocab_path'])

    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    dataset = MSCOCODataset(
        image_dir=config['dataset']['image_dir'],
        captions_path=config['dataset']['captions_path'],
        tokenizer=tokenizer,
        transform=transform,
        max_len=config['model']['max_len']
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=0  # ✅ Fixed for Windows
    )

    # Model
    model = ImageCaptioningModel(
        vocab_size=len(tokenizer.word2idx),
        embed_size=config['model']['embed_size'],
        decoder_dim=config['model']['decoder_dim'],
        attention_dim=config['model']['attention_dim'],
        dropout=config['model']['dropout'],
        max_len=config['model']['max_len']
    ).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.word2idx["<PAD>"])
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(config['train']['learning_rate']),
                                  weight_decay=float(config['train']['weight_decay']))

    # Training loop
    best_loss = float('inf')
    for epoch in range(config['train']['num_epochs']):
        model.train()
        total_loss = 0

        for i, (images, captions) in enumerate(dataloader):
            images, captions = images.to(device), captions.to(device)

            optimizer.zero_grad()

            outputs = model(images, captions[:, :-1])  # input everything except last token
            targets = captions[:, 1:]                  # predict everything except <START>

            outputs = outputs.reshape(-1, outputs.shape[2])
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)
            loss.backward()

            if config['train']['clip_grad']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['train']['clip_grad'])

            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % config['train']['log_interval'] == 0:
                print(f"Epoch [{epoch+1}/{config['train']['num_epochs']}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} complete. Avg Loss: {avg_loss:.4f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs(config['train']['save_dir'], exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config['train']['save_dir'], "best_model.pth"))
            print("✅ Best model saved.")

# ✅ Required for multiprocessing on Windows
if __name__ == "__main__":
    main()
