from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"

SEN_LEN = 128
BATCH_SIZE = 64
NUM_WORKERS = 0
EMBEDDING_DIM = 128
PADDING_IDX = 0
HIDDEN_DIM = 256
LEARNING_RATE = 1e-3
EPOCHS = 10