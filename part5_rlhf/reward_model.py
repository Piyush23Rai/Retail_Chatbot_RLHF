import torch
import torch.nn as nn
import torch.nn.functional as F
import random


# =====================================================
# Reward Model
# =====================================================

class RewardModel(nn.Module):
    """
    Reward model built on top of SFT transformer backbone.
    Outputs scalar reward.
    """

    def __init__(self, sft_model):
        super().__init__()

        self.token_embedding = sft_model.token_embedding
        self.pos_embedding = sft_model.pos_embedding
        self.transformer = sft_model.transformer

        embed_dim = sft_model.token_embedding.embedding_dim
        self.reward_head = nn.Linear(embed_dim, 1)

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.size()

        positions = torch.arange(
            0, seq_len, device=input_ids.device
        ).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_embedding(input_ids) + self.pos_embedding(positions)

        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device),
            diagonal=1
        ).bool()

        padding_mask = (input_ids == 0)

        x = self.transformer(
            x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        last_hidden = x[:, -1, :]
        reward = self.reward_head(last_hidden)

        return reward.squeeze(-1)


# =====================================================
# Preference Pair Creation
# =====================================================

def create_preference_pairs(dataset, n_pairs=1000):

    positives = [d for d in dataset if d["high_quality"] == 1]
    negatives = [d for d in dataset if d["high_quality"] == 0]

    pairs = []

    for _ in range(n_pairs):
        pos = random.choice(positives)
        neg = random.choice(negatives)
        pairs.append((pos, neg))

    return pairs


# =====================================================
# Bradley-Terry Loss
# =====================================================

def preference_loss(r_pos, r_neg):
    return -F.logsigmoid(r_pos - r_neg).mean()


# =====================================================
# Train Reward Model
# =====================================================

def train_reward_model(
    reward_model,
    dataset,
    tokenizer,
    device,
    epochs=5,
    n_pairs=1000,
    lr=1e-4
):

    reward_model.train()

    pairs = create_preference_pairs(dataset, n_pairs)

    optimizer = torch.optim.Adam(reward_model.parameters(), lr=lr)

    for epoch in range(epochs):

        total_loss = 0

        for pos, neg in pairs:

            pos_text = pos["context"] + " " + pos["response"]
            neg_text = neg["context"] + " " + neg["response"]

            pos_ids = tokenizer(pos_text).to(device)
            neg_ids = tokenizer(neg_text).to(device)

            r_pos = reward_model(pos_ids)
            r_neg = reward_model(neg_ids)

            loss = preference_loss(r_pos, r_neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Reward Epoch {epoch+1} | Loss: {total_loss:.4f}")

    return reward_model


# =====================================================
# Evaluate Reward Model Accuracy
# =====================================================

def evaluate_reward_model(reward_model, dataset, tokenizer, device, n_samples=200):

    reward_model.eval()

    pairs = create_preference_pairs(dataset, n_samples)

    correct = 0

    with torch.no_grad():

        for pos, neg in pairs:

            pos_text = pos["context"] + " " + pos["response"]
            neg_text = neg["context"] + " " + neg["response"]

            pos_ids = tokenizer(pos_text).to(device)
            neg_ids = tokenizer(neg_text).to(device)

            r_pos = reward_model(pos_ids)
            r_neg = reward_model(neg_ids)

            if r_pos > r_neg:
                correct += 1

    accuracy = correct / n_samples

    print(f"Reward Model Accuracy: {accuracy:.4f}")

    return accuracy
