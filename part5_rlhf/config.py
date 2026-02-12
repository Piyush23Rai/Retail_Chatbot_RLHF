from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SFT_MODEL_PATH = BASE_DIR.parent / "part4_sft" / "sft_model.pt"
SFT_DATA_PATH = BASE_DIR.parent / "part4_sft" / "sft_data.json"
VOCAB_PATH = BASE_DIR.parent / "part3_pretraining" / "vocab.json"
POLICY_LOSS_CURVE_PATH = BASE_DIR.parent / "part5_rlhf" / "policy_loss.png"
RLHF_MODEL_PATH = BASE_DIR.parent / "part5_rlhf" / "rlhf_model.pt"
REWARD_CURVE_PATH = BASE_DIR.parent / "part5_rlhf" /"reward_curve.png"
KL_CURVE_PATH = BASE_DIR.parent / "part5_rlhf" / "kl_curve.png"
TRAINING_CURVE_PATH = BASE_DIR.parent / "part5_rlhf" / "training_curve.png"