import os
import json
import random
import re
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
    return re.findall(r"\b\w+\b", text.lower())


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

def get_base_pairs():
    return {
        # Product recommendation
        "Customer wants durable jeans for hiking":
            "I recommend our TrekPro Jeans rated 4.8/5 with 6-month warranty.",

        "Customer needs waterproof jacket for trekking":
            "Our StormShield Jacket is waterproof, lightweight, and rated 4.7/5.",

        "Customer looking for running shoes":
            "Our SprintMax shoes provide comfort and durability for long runs.",

        # Budget concerns
        "Customer looking for budget shoes":
            "We have affordable options starting at $29.99.",

        "Customer wants cheapest backpack":
            "Our BasicTrail backpack offers great value at $24.99.",

        # Eco-friendly
        "Customer wants eco-friendly products":
            "Our eco-line products are sustainably sourced and environmentally friendly.",

        "Customer prefers recycled materials":
            "We offer products made from recycled and biodegradable materials.",

        # Complaint handling
        "Customer had previous bad experience":
            "I understand your frustration. Let me help you find a better option.",

        "Customer received damaged item":
            "I apologize for the inconvenience. We will arrange a replacement immediately.",

        # Returns
        "Customer asking about return policy":
            "We offer 30-day returns for unused items with original packaging.",

        "Customer wants refund process details":
            "Refunds are processed within 5 business days after we receive the item.",
    }


def create_sft_dataset():
    """
    Generate 5,000 (context, response) pairs
    Manually label 500 as high-quality
    """


    data = []

    base_pairs = get_base_pairs()
    contexts = list(base_pairs.keys())
    

    for i in range(SFT_DATA_SIZE):
        context = random.choice(contexts)
        response = base_pairs[context]

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

    # Use only high-quality examples
    filtered_data = [d for d in sft_data if d["high_quality"] == 1]

    print(f"Training on {len(filtered_data)} high-quality examples")

    dataset = SFTDataset(filtered_data, vocab)
    
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
def generate_response(model, context, vocab, max_len=50, temperature=TEMPERATURE):

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
        # Prevent PAD token generation
        probs[vocab["<PAD>"]] = 0
        probs = probs / probs.sum()
        
        next_token = torch.multinomial(probs, 1).item()

        if next_token == vocab["<EOS>"]:
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

def get_test_data():

    return [
        {"context": "Customer wants eco-friendly products"},
        {"context": "Customer has received a damaged item"}
    ]

def qualitative_examples(pretrained_model, sft_model, test_data, vocab):

    print("\n--- QUALITATIVE ANALYSIS ---\n")

    for item in test_data:

        context = item["context"]

        pre_output = generate_response(pretrained_model, context, vocab, temperature=TEMPERATURE)
        sft_output = generate_response(sft_model, context, vocab, temperature=TEMPERATURE)

        print("Context:", context)
        print("Pretrained:", pre_output)
        print("SFT:", sft_output)
        print("-" * 60)


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

    print("Examples..")
    qualitative_examples(
        pretrained_model=pretrained_model,
        sft_model=sft_model,
        test_data=get_test_data(),
        vocab=vocab
    )


if __name__ == "__main__":
    main()
