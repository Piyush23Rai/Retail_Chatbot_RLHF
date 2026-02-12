import json
import copy
import torch
from .reward_model import *
from .ppo import *
from part3_pretraining.pretraining import SimpleTransformer
from .config import *
from .utils import load_sft_data
from .utils import Tokenizer
from torch.utils.data import DataLoader

tokenizer = Tokenizer(VOCAB_PATH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load vocab from Part 3
with open(VOCAB_PATH, "r") as f:
    vocab = json.load(f)

vocab_size = len(vocab)

# Load SFT model
sft_model = SimpleTransformer(vocab_size=vocab_size)
sft_model.load_state_dict(torch.load(SFT_MODEL_PATH))

# Load SFT dataset
sft_dataset = load_sft_data(SFT_DATA_PATH)

# Train reward model
reward_model = RewardModel(sft_model).to(device)
reward_model = train_reward_model(
    reward_model,
    sft_dataset,
    tokenizer.encode,
    device,
    epochs=5
)

# Evaluate reward model
evaluate_reward_model(
    reward_model,
    sft_dataset,
    tokenizer.encode,
    device
)

# =====================================================
# PPO Training
# =====================================================

policy_model = copy.deepcopy(sft_model).to(device)
reference_model = copy.deepcopy(sft_model).to(device)

for p in reference_model.parameters():
    p.requires_grad = False

contexts = [d["context"] for d in sft_dataset]
dataloader = DataLoader(contexts, batch_size=16, shuffle=True)

reward_history = train_ppo(
    policy_model=policy_model,
    reference_model=reference_model,
    reward_model=reward_model,
    data_loader=dataloader,
    tokenizer=tokenizer.encode,
    device=device,
    epochs=10
)


# =====================================================
# Save Model
# =====================================================

torch.save(policy_model.state_dict(), RLHF_MODEL_PATH)


# =====================================================
# Plot Reward Curve
# =====================================================

plt.plot(reward_history)
plt.xlabel("Epoch")
plt.ylabel("Average Reward")
plt.title("PPO Training Reward")
plt.savefig(TRAINING_CURVE_PATH)

print("\nRLHF Training Complete.")
