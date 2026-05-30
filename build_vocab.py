# build_vocab.py

import json
from utils.tokenizer import Tokenizer

# Load the 1000-caption subset
with open("data/raw/captions_subset.json", 'r') as f:
    data = json.load(f)

captions = [ann["caption"] for ann in data["annotations"]]

# Build and save vocab
tokenizer = Tokenizer(freq_threshold=3)  # or 1 if dataset is small
tokenizer.build_vocab(captions)
tokenizer.save_vocab("data/vocab.pkl")

print(f"✅ Vocabulary built. Total words: {len(tokenizer.word2idx)}")
