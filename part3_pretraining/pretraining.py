# part3_pretraining/pretraining.py
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from .config import *




# ===============================
# SimpleTransformer
# ===============================

class SimpleTransformer(nn.Module):
    """
    Proper autoregressive Transformer Language Model
    """

    def __init__(self, vocab_size):
        super().__init__()

        self.vocab_size = vocab_size

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM)

        # Positional embedding
        self.pos_embedding = nn.Embedding(MAX_SEQ_LEN, EMBED_DIM)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            dropout=DROPOUT,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        # Output projection
        self.fc_out = nn.Linear(EMBED_DIM, vocab_size)

    def forward(self, x):
        """
        x: (batch_size, seq_len)
        """

        batch_size, seq_len = x.size()

        # Create position indices
        positions = torch.arange(
            0, seq_len, device=x.device
        ).unsqueeze(0).expand(batch_size, seq_len)

        # Token + positional embeddings
        x = self.token_embedding(x) + self.pos_embedding(positions)

        # ==============================
        # Causal Mask (prevents looking ahead)
        # ==============================
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        # ==============================
        # Padding Mask
        # ==============================
        padding_mask = (x[:, :, 0] == 0)  # detect PAD via token index 0
        # Better approach below (see note)

        # Actually better padding mask:
        # padding_mask = (input_ids == pad_idx)
        # But since we don't pass pad_idx here,
        # we assume <PAD> index is 0.

        # Transformer
        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        logits = self.fc_out(x)

        return logits



# ===============================
# Dataset
# ===============================
class ConversationDataset(Dataset):
    def __init__(self, texts, word2idx, max_len=256):
        self.texts = texts
        self.word2idx = word2idx
        self.max_len = max_len

        self.pad_idx = self.word2idx["<PAD>"]
        self.unk_idx = self.word2idx["<UNK>"]

    def encode(self, text):
        tokens = text.split()
        ids = [self.word2idx.get(t, self.unk_idx) for t in tokens]

        # Append EOS
        ids.append(self.word2idx["<EOS>"])

        # Truncate
        ids = ids[: self.max_len]

        # Ensure minimum length of 2 (needed for x/y)
        if len(ids) < 2:
            ids = ids + [self.pad_idx]

        return ids

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.encode(self.texts[idx])

        x = torch.tensor(encoded[:-1], dtype=torch.long)
        y = torch.tensor(encoded[1:], dtype=torch.long)

        return x, y



# =================================================
# Padding function to have equal length batch size
# =================================================
def collate_fn(batch):
    """
    Pads sequences in the batch to the same length
    """
    xs, ys = zip(*batch)

    max_len = max(x.size(0) for x in xs)

    padded_xs = []
    padded_ys = []

    for x, y in zip(xs, ys):
        pad_len = max_len - x.size(0)

        padded_x = torch.cat(
            [x, torch.zeros(pad_len, dtype=torch.long)]
        )
        padded_y = torch.cat(
            [y, torch.zeros(pad_len, dtype=torch.long)]
        )

        padded_xs.append(padded_x)
        padded_ys.append(padded_y)

    return torch.stack(padded_xs), torch.stack(padded_ys)



# ===============================
# prepare_pretraining_data 
# ===============================

def prepare_pretraining_data():
    """
    Loads cleaned conversations, builds vocabulary,
    and prepares train/val/test dataloaders
    """
    df = pd.read_csv(DATA_PATH)

    assert df["conversation_text"].isna().sum() == 0
    assert df["conversation_text"].apply(lambda x: isinstance(x, str)).all()

    texts = df["conversation_text"].tolist()

    # -------------------------
    # Train / Val / Test split
    # -------------------------
    train_texts, temp_texts = train_test_split(
        texts, test_size=0.2, random_state=42
    )

    val_texts, test_texts = train_test_split(
        temp_texts, test_size=0.5, random_state=42
    )

    # -------------------------
    # Build vocabulary (word-level)
    # -------------------------
    counter = Counter()
    for t in texts:
        counter.update(t.split())

    vocab = ["<PAD>", "<UNK>", "<EOS>"] + list(counter.keys())
    word2idx = {w: i for i, w in enumerate(vocab)}

    # save vocabulary (to be used during SFT)
    with open(VOCAB_PATH, "w") as f:
        json.dump(word2idx, f)

    # -------------------------
    # Datasets
    # -------------------------
    train_dataset = ConversationDataset(
        train_texts, word2idx, max_len=MAX_SEQ_LEN
    )

    val_dataset = ConversationDataset(
        val_texts, word2idx, max_len=MAX_SEQ_LEN
    )

    test_dataset = ConversationDataset(
        test_texts, word2idx, max_len=MAX_SEQ_LEN
    )

    # -------------------------
    # DataLoaders
    # -------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, test_loader, word2idx




# ===============================
# compute_perplexity 
# ===============================

def compute_perplexity(val_loss):
    return math.exp(val_loss)


# ===============================
# train_pretrained_model (REQUIRED)
# ===============================

def train_pretrained_model(epochs=EPOCHS):
    # -------------------------
    # Data
    # -------------------------
    train_loader, val_loader, test_loader, word2idx = prepare_pretraining_data()
    vocab_size = len(word2idx)
    pad_idx = word2idx["<PAD>"]

    # -------------------------
    # Model
    # -------------------------
    model = SimpleTransformer(vocab_size=vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    train_losses = []
    val_losses = []

    # -------------------------
    # Training loop
    # -------------------------
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0

        for x, y in train_loader:
            optimizer.zero_grad()
            logits = model(x)

            loss = criterion(
                logits.view(-1, vocab_size),
                y.view(-1)
            )

            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        epoch_val_loss = 0.0

        with torch.no_grad():
            for x, y in val_loader:
                logits = model(x)
                loss = criterion(
                    logits.view(-1, vocab_size),
                    y.view(-1)
                )
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        val_ppl = compute_perplexity(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val PPL: {val_ppl:.2f}"
        )

    # -------------------------
    # Save artifacts
    # -------------------------
    plot_loss(train_losses, val_losses)
    save_model(model)

    return model



# ===============================
# save_model / load_model (REQUIRED)
# ===============================

def save_model(model, path=MODEL_PATH):
    torch.save(model.state_dict(), path)


def load_model(model, path=MODEL_PATH):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


# ===============================
# Plotting
# ===============================
def plot_loss(train_losses, val_losses):
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Pretraining Loss Curve")
    plt.savefig(LOSS_PLOT_PATH)
    plt.close()



# ===============================
# Entry Point
# ===============================

if __name__ == "__main__":
    model = train_pretrained_model(epochs=5)
    save_model(model)
