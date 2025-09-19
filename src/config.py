# Global constants and default hyperparameters

# Number of letters in the English alphabet.
N = 26  # we use indices 1..26 for 'a'..'z'; 0 is the blank ' '

# ===== Default configs for the fixed-delay task =====
DELAY = 4
DATASET_SIZE = 200_000
HIDDEN_SIZE_ECHO = 128
BATCH_SIZE_ECHO = 64
EPOCHS_ECHO = 5
LR_ECHO = 1e-3

# ===== Default configs for the variable-delay task =====
MAX_DELAY = 12
SEQ_LENGTH_VAR = 20
HIDDEN_SIZE_VAR = 64
BATCH_SIZE_VAR = 32
DELAY_EMBED_DIM = 16
POS_EMBED_DIM = 8
EPOCHS_VAR = 8
LR_VAR = 1e-2
STEP_SIZE_VAR = 10
GAMMA_VAR = 0.1

# Test constraints (like your original asserts)
MAX_TOTAL_TRAIN_SECONDS = 600  # 10 minutes
MIN_ACC_ECHO = 0.99
MIN_ACC_VAR = 0.95
