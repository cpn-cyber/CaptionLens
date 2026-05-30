import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from utils.tokenizer import Tokenizer

class MSCOCODataset(Dataset):
    def __init__(self, image_dir, captions_path, tokenizer, transform=None, max_len=50):
        self.image_dir = image_dir
        self.captions_path = captions_path
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len

        with open(captions_path, 'r') as f:
            data = json.load(f)

        self.images = []
        self.captions = []

        for annot in data['annotations']:
            caption = annot['caption']
            image_id = annot['image_id']
            image_filename = f"{image_id:012d}.jpg"

            self.images.append(image_filename)
            self.captions.append(caption)

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        # Tokenize caption
        tokens = [self.tokenizer.word2idx["<START>"]]
        tokens += self.tokenizer.numericalize(caption)
        tokens += [self.tokenizer.word2idx["<END>"]]

        if len(tokens) < self.max_len:
            tokens += [self.tokenizer.word2idx["<PAD>"]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        caption_tensor = torch.tensor(tokens, dtype=torch.long)

        return image, caption_tensor
