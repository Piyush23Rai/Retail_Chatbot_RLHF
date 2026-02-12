import re
import torch
import json

def load_sft_data(path="../part4_sft/sft_dataset.json"):
    """
    Load SFT dataset generated in Part 4.
    Returns list of dicts:
    {
        "context": "...",
        "response": "...",
        "high_quality": 0 or 1
    }
    """

    with open(path, "r") as f:
        data = json.load(f)

    return data

class Tokenizer:

    def __init__(self, vocab_path, max_len=30):

        with open(vocab_path, "r") as f:
            self.stoi = json.load(f)

        self.max_len = max_len
        self.pad_idx = 0
        self.unk_idx = 1

    # SAME tokenization used in SFT
    def tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def encode(self, text):

        tokens = self.tokenize(text)

        ids = [self.stoi.get(token, self.unk_idx) for token in tokens]

        ids = ids[:self.max_len]
        ids += [self.pad_idx] * (self.max_len - len(ids))

        return torch.tensor(ids).unsqueeze(0)

