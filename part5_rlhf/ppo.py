# part5_rlhf/ppo.py

import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from .config import *


# ==========================================
# 1️⃣ Advantage Estimation
# ==========================================

def compute_advantages(rewards, values, gamma=0.99):

    returns = rewards  # episodic reward (no multi-step rollout)

    advantages = returns - values

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return advantages



# ==========================================
# 2️⃣ PPO Clipped Objective
# ==========================================

def ppo_loss(old_logprobs,
             new_logprobs,
             advantages,
             epsilon=0.2):

    log_ratio = torch.clamp(new_logprobs - old_logprobs, -10, 10)
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages

    return -torch.min(unclipped, clipped).mean()



# ==========================================
# 3️⃣ KL Divergence
# ==========================================

def compute_kl_divergence(old_logprobs, new_logprobs):
    return torch.mean(old_logprobs - new_logprobs)


# ==========================================
# 4️⃣ Compute Sequence Log Probability
# ==========================================

def compute_sequence_logprob(model, input_ids):

    logits = model(input_ids)
    logprobs = torch.log_softmax(logits, dim=-1)

    selected = logprobs.gather(
        -1,
        input_ids.unsqueeze(-1)
    ).squeeze(-1)

    # USE MEAN NOT SUM
    return selected.mean(dim=1)



# ==========================================
# 5️⃣ PPO Training
# ==========================================

def train_ppo(policy_model,
              reference_model,
              reward_model,
              data_loader,
              tokenizer,
              device,
              epochs=5,
              epsilon=0.2,
              beta=0.01,
              lr=1e-5):

    optimizer = optim.Adam(policy_model.parameters(), lr=lr)

    policy_losses = []
    rewards_track = []
    kl_track = []

    policy_model.train()
    reference_model.eval()
    reward_model.eval()

    for epoch in range(epochs):

        total_policy_loss = 0
        total_reward = 0
        total_kl = 0

        for contexts in data_loader:

            # # tokenize batch (batch_size=1 assumed)
            # input_ids = tokenizer(context[0]).to(device)
            batch_ids = []

            for text in contexts:
                ids = tokenizer(text)   # should return [1, seq_len]
                batch_ids.append(ids)

            input_ids = torch.cat(batch_ids, dim=0).to(device)

            # ---- Old policy log probs (reference)
            with torch.no_grad():
                old_logprob = compute_sequence_logprob(
                    reference_model,
                    input_ids
                )

            # ---- New policy log probs
            new_logprob = compute_sequence_logprob(
                policy_model,
                input_ids
            )

            # ---- Reward
            with torch.no_grad():
                reward = reward_model(input_ids)

            baseline = reward.mean().detach()
            advantages = compute_advantages(reward, baseline)

            # print("Input IDs shape:", input_ids.shape)
            # print("Any NaN in input_ids?", torch.isnan(input_ids).any())

            # print("Old logprob NaN?", torch.isnan(old_logprob).any())
            # print("New logprob NaN?", torch.isnan(new_logprob).any())

            # print("Reward NaN?", torch.isnan(reward).any())


            # ---- PPO loss
            policy_loss = ppo_loss(
                old_logprob,
                new_logprob,
                advantages,
                epsilon
            )

            # ---- KL penalty
            kl = compute_kl_divergence(old_logprob, new_logprob)

            loss = policy_loss + beta * kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_reward += reward.mean().item()
            total_kl += kl.item()

        avg_loss = total_policy_loss / len(data_loader)
        avg_reward = total_reward / len(data_loader)
        avg_kl = total_kl / len(data_loader)

        policy_losses.append(avg_loss)
        rewards_track.append(avg_reward)
        kl_track.append(avg_kl)

        print(f"\n[PPO] Epoch {epoch+1}")
        print("Policy Loss:", round(avg_loss, 4))
        print("Reward:", round(avg_reward, 4))
        print("KL:", round(avg_kl, 6))

    # Save final model
    torch.save(policy_model.state_dict(), RLHF_MODEL_PATH)

    # ==========================================
    # Plot training curves
    # ==========================================

    plt.figure()
    plt.plot(policy_losses)
    plt.title("Policy Loss")
    plt.savefig(POLICY_LOSS_CURVE_PATH)

    plt.figure()
    plt.plot(rewards_track)
    plt.title("Reward")
    plt.savefig(REWARD_CURVE_PATH)

    plt.figure()
    plt.plot(kl_track)
    plt.title("KL Divergence")
    plt.savefig(KL_CURVE_PATH)

    return policy_losses, rewards_track, kl_track


# ==========================================
# 6️⃣ Evaluation
# ==========================================

def evaluate_rlhf_policy(policy_model,
                         reward_model,
                         data_loader,
                         tokenizer,
                         device):

    total_reward = 0

    policy_model.eval()
    reward_model.eval()

    with torch.no_grad():
        for context in data_loader:

            input_ids = tokenizer(context[0]).to(device)
            reward = reward_model(input_ids)

            total_reward += reward.mean().item()

    avg_reward = total_reward / len(data_loader)

    print("Average Reward per Episode:", round(avg_reward, 4))

    return avg_reward
