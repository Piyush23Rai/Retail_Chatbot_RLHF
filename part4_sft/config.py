import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ===============================
# Configuration
# ===============================

SFT_DATA_SIZE = 5000
MANUAL_LABEL_SIZE = 500
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_LEN = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PRETRAINED_PATH = BASE_DIR.parent / "part3_pretraining" / "pretrained_model.pt"
SFT_MODEL_PATH = BASE_DIR.parent / "part4_sft" / "sft_model.pt"
SFT_DATA_PATH = BASE_DIR.parent / "part4_sft" / "sft_data.json"
LOSS_CURVE_PATH = BASE_DIR.parent / "part4_sft" / "sft_loss_curve.png"
EVAL_PATH = BASE_DIR.parent / "part4_sft" / "evaluation.json"
VOCAB_PATH = BASE_DIR.parent / "part3_pretraining" / "vocab.json"