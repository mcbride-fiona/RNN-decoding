import argparse
import time
import torch
from torch.utils.data import DataLoader

from .config import (
    MAX_DELAY, SEQ_LENGTH_VAR,
    HIDDEN_SIZE_VAR, BATCH_SIZE_VAR, EPOCHS_VAR, LR_VAR
)
from .datasets import VariableDelayEchoDatasetBalanced as VariableDelayEchoDataset
from .model import VariableDelayGRUMemory
from .train import train_variable_delay
from .eval import test_model_variable_delay, assert_runtime, assert_accuracy_variable

def main():
    parser = argparse.ArgumentParser(description="Variable-delay Echo with GRU")
    parser.add_argument("--max-delay", type=int, default=MAX_DELAY)
    parser.add_argument("--seq-length", type=int, default=SEQ_LENGTH_VAR)
    parser.add_argument("--size", type=int, default=60_000)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE_VAR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_VAR)
    parser.add_argument("--epochs", type=int, default=EPOCHS_VAR)
    parser.add_argument("--lr", type=float, default=LR_VAR)
    args = parser.parse_args()

    start = time.time()

    ds = VariableDelayEchoDataset(max_delay=args.max_delay, seq_length=args.seq_length, size=args.size)
    dl = DataLoader(ds, batch_size=args.batch_size)

    model = VariableDelayGRUMemory(hidden_size=args.hidden_size, max_delay=args.max_delay)
    train_variable_delay(model, dl, epochs=args.epochs, lr=args.lr)

    acc = test_model_variable_delay(model, max_delay=args.max_delay)
    print(f"Accuracy (variable max_delay={args.max_delay}): {acc:.4f}")

    assert_runtime(start)
    assert_accuracy_variable(acc)
    print("tests passed")

if __name__ == "__main__":
    main()

