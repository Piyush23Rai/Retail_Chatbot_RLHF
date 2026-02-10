from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ===============================
# Configuration (helper-like)
# ===============================

EMBED_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
DROPOUT = 0.1

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3
MAX_SEQ_LEN = 50

GAMMA = 0.99  # discount factor (conceptual MDP framing)

DATA_PATH = BASE_DIR.parent / "part1_dq_ddq" / "cleaned_conv.csv"
MODEL_PATH = BASE_DIR.parent / "part3_pretraining" / "pretrained_model.pt"
LOSS_PLOT_PATH = BASE_DIR.parent / "part3_pretraining" / "loss_curves.png"