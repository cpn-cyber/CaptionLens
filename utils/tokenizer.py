import re
import pickle
from collections import Counter

class Tokenizer:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {
            "<PAD>": 0,
            "<START>": 1,
            "<END>": 2,
            "<UNK>": 3
        }
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}

    def clean_caption(self, caption):
        caption = caption.lower()
        caption = re.sub(r"[^a-z0-9\s]", "", caption)
        caption = re.sub(r"\s+", " ", caption).strip()
        return caption

    def build_vocab(self, sentence_list):
        frequencies = Counter()
        idx = 4  # Start after special tokens

        for sentence in sentence_list:
            sentence = self.clean_caption(sentence)
            for word in sentence.split(' '):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1

    def numericalize(self, caption):
        caption = self.clean_caption(caption)
        tokens = caption.split(' ')
        return [
            self.word2idx.get(word, self.word2idx["<UNK>"])
            for word in tokens
        ]

    def decode(self, idx_list):
        return " ".join([self.idx2word.get(idx, "<UNK>") for idx in idx_list])

    def save_vocab(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({
                "word2idx": self.word2idx,
                "idx2word": self.idx2word
            }, f)

    def load_vocab(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.word2idx = data["word2idx"]
            self.idx2word = data["idx2word"]
