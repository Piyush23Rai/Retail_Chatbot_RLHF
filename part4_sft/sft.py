import os
import json
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from .config import *

from nltk.translate.bleu_score import SmoothingFunction
smooth = SmoothingFunction().method1


# ===============================
# Utilities
# ===============================

def tokenize(text):
    return text.lower().split()


def encode(text, vocab):
    tokens = tokenize(text)
    ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens]
    return ids[:MAX_LEN]


def pad_sequence(seq, max_len):
    return seq + [0] * (max_len - len(seq))


# Load vocab from Part 3
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

vocab_size = len(vocab)


# ===============================
# Dataset
# ===============================

class SFTDataset(Dataset):
    def __init__(self, pairs, vocab):
        self.pairs = pairs
        self.vocab = vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        context = self.pairs[idx]["context"]
        response = self.pairs[idx]["response"]

        context_ids = encode(context, self.vocab)
        response_ids = encode(response, self.vocab)

        input_ids = context_ids + response_ids
        input_ids = input_ids[:MAX_LEN]

        input_ids = pad_sequence(input_ids, MAX_LEN)

        # Targets
        labels = input_ids.copy()

        # Mask context tokens in loss
        for i in range(len(context_ids)):
            labels[i] = -100   # ignore index for CrossEntropyLoss

        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(labels[1:], dtype=torch.long)

        return x, y



# ===============================
# 1️⃣ Create SFT Dataset
# ===============================

def create_sft_dataset():
    """
    Generate 5,000 (context, response) pairs
    Manually label 500 as high-quality
    """

    # Simulated synthetic contexts
    sample_contexts = [
        "Customer wants durable jeans for hiking",
        "Customer had previous bad experience",
        "Customer asking about return policy",
        "Customer looking for budget shoes",
        "Customer wants eco-friendly products",
    ]

    sample_responses = [
        "I recommend our TrekPro Jeans rated 4.8/5 with 6-month warranty.",
        "I understand your frustration. Let me help you find a better option.",
        "We offer 30-day returns for unused items.",
        "We have affordable options starting at $29.99.",
        "Our eco-line products are sustainably sourced.",
    ]

    data = []

    for i in range(SFT_DATA_SIZE):
        context = random.choice(sample_contexts)
        response = random.choice(sample_responses)

        quality = 1 if i < MANUAL_LABEL_SIZE else 0

        data.append({
            "context": context,
            "response": response,
            "high_quality": quality
        })

    with open(SFT_DATA_PATH, "w") as f:
        json.dump(data, f, indent=4)

    print(f"SFT dataset saved to {SFT_DATA_PATH}")
    return data


# ===============================
# 2️⃣ Fine-Tuning
# ===============================

def train_sft(pretrained_model, sft_data, epochs=EPOCHS):

    # texts = [d["context"] + " " + d["response"] for d in sft_data]
    # vocab = build_vocab(texts)

    dataset = SFTDataset(sft_data, vocab)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = pretrained_model
    model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    train_losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(x)

            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        train_losses.append(avg_loss)

        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), SFT_MODEL_PATH)
    print("SFT model saved.")

    plt.plot(train_losses)
    plt.title("SFT Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()

    return model


# ===============================
# Text Generation
# ===============================
def generate_response(model, context, vocab, max_len=50, temperature=0.8):

    model.eval()
    idx2word = {idx: word for word, idx in vocab.items()}

    input_ids = encode(context, vocab)
    generated = input_ids.copy()

    for _ in range(max_len):

        input_tensor = torch.tensor(
            pad_sequence(generated, MAX_LEN),
            dtype=torch.long
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)

        next_token_logits = logits[0, len(generated)-1] / temperature
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        if next_token == vocab["<PAD>"]:
            break

        generated.append(next_token)

    words = [idx2word.get(tok, "<UNK>") for tok in generated[len(input_ids):]]
    return " ".join(words)




# ===============================
# 3️⃣ Metrics (Real Generation)
# ===============================

def compute_instruction_following_metrics(model, test_data, vocab):

    model.eval()
    bleu_scores = []
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    with torch.no_grad():
        for item in test_data[:200]:

            context = item["context"]
            reference = item["response"]

            generated = generate_response(model, context, vocab)
            # print("Context:", context)
            # print("Reference:", reference)
            # print("Generated:", generated)
            # print("-" * 50)

            bleu = sentence_bleu(
                [reference.split()],
                generated.split(),
                smoothing_function=smooth
            )

            rouge_score = rouge.score(reference, generated)["rougeL"].fmeasure

            bleu_scores.append((bleu, rouge_score))

    avg_bleu = sum(b[0] for b in bleu_scores) / len(bleu_scores)
    avg_rouge = sum(b[1] for b in bleu_scores) / len(bleu_scores)

    return avg_bleu, avg_rouge



# ===============================
# 4️⃣ Evaluation Comparison
# ===============================

def evaluate_sft_vs_pretrained(pretrained_model, sft_model, test_data, vocab):

    bleu_pre, rouge_pre = compute_instruction_following_metrics(
        pretrained_model, test_data, vocab
    )

    bleu_sft, rouge_sft = compute_instruction_following_metrics(
        sft_model, test_data, vocab
    )

    results = {
        "pretrained": {
            "BLEU": bleu_pre,
            "ROUGE": rouge_pre
        },
        "sft": {
            "BLEU": bleu_sft,
            "ROUGE": rouge_sft
        }
    }

    with open(EVAL_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("Evaluation saved to evaluation.json")

    return results


# ===============================
# Main Pipeline
# ===============================

def main():

    print("Creating SFT dataset...")
    sft_data = create_sft_dataset()

    print("Loading pretrained model...")
    from part3_pretraining.pretraining import SimpleTransformer

    pretrained_model = SimpleTransformer(vocab_size=vocab_size)
    pretrained_model.load_state_dict(torch.load(PRETRAINED_PATH))

    print("Fine-tuning model...")
    sft_model = train_sft(pretrained_model, sft_data)

    print("Evaluating...")
    results = evaluate_sft_vs_pretrained(
        pretrained_model,
        sft_model,
        sft_data,
        vocab
    )

    print("Results:", results)


if __name__ == "__main__":
    main()
