import json
import os

import yaml

from utils.tokenizer import Tokenizer


with open("config/config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

captions_path = config["dataset"]["train_captions_path"]
vocab_path = config["processed_data"]["vocab_path"]

if not os.path.exists(captions_path):
    raise FileNotFoundError(
        f"Training captions not found: {captions_path}. "
        "Run prepare_splits.py first."
    )

with open(captions_path, "r", encoding="utf-8") as f:
    data = json.load(f)

captions = [ann["caption"] for ann in data["annotations"]]

tokenizer = Tokenizer(freq_threshold=3)
tokenizer.build_vocab(captions)
os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
tokenizer.save_vocab(vocab_path)

print(f"Vocabulary built from: {captions_path}")
print(f"Vocabulary saved to: {vocab_path}")
print(f"Total words: {len(tokenizer.word2idx)}")
