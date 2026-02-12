
# Part 5: Reinforcement Learning from Human Feedback (RLHF)

## Overview

This module implements Reinforcement Learning from Human Feedback (RLHF) using Proximal Policy Optimization (PPO) to align the chatbot policy with human preferences.

The objective is to improve the Supervised Fine-Tuned (SFT) model using a learned reward model while ensuring the updated policy does not drift excessively from the reference model.

---

## Objectives

- Use SFT model as reference policy (π_old)
- Use trained reward model as reward signal
- Implement PPO with advantage estimation
- Train for 10 epochs with batch_size=16
- Track:
  - Policy loss
  - Average reward
  - KL divergence from reference policy
- Ensure KL divergence remains < 0.1

---

## Folder Structure

part5_rlhf/
├── reward_model.py
├── ppo.py
├── run_rlhf.py
├── rlhf_model.pt
├── policy_loss.png
├── reward_curve.png
├── kl_curve.png
└── README.md

---

## How to Execute Part 5

From the project root directory:

    python -m part5_rlhf.run_rlhf

Note: Part 4 - Should be executed before Part 5

This will:
1. Load the SFT model as reference policy
2. Train the reward model
3. Train PPO for 10 epochs
4. Save the final aligned model (rlhf_model.pt)
5. Generate training curves

---

## PPO Mathematical Formulation

Clipped Objective:

L^CLIP = E[min(r_t(θ) Â_t, clip(r_t(θ), 1-ε, 1+ε) Â_t)]

Where:

r_t(θ) = π_new(a|s) / π_old(a|s)

KL divergence is monitored to ensure policy stability:

KL(π_new || π_old) < 0.1

---

## Observations

- Policy loss stabilizes across epochs
- KL divergence remains below threshold
- Policy improves alignment without catastrophic drift

---

## Retail Impact

RLHF ensures:
- More empathetic complaint handling
- Better refund management responses
- Improved customer satisfaction
- Safer deployment in retail environments
