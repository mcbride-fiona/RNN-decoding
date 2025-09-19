import argparse
import time
import torch
from torch.utils.data import DataLoader

from .config import (
    N, DELAY, DATASET_SIZE,
    HIDDEN_SIZE_ECHO, BATCH_SIZE_ECHO, EPOCHS_ECHO, LR_ECHO
)
from .datasets import EchoDataset
from .model import GRUMemory
from .train import train_fixed_delay
from .eval import test_model_fixed_delay, assert_runtime, assert_accuracy_echo

def main():
    parser = argparse.ArgumentParser(description="Fixed-delay Echo with GRU")
    parser.add_argument("--delay", type=int, default=DELAY)
    parser.add_argument("--size", type=int, default=DATASET_SIZE)
    parser.add_argument("--seq-length", type=int, default=15)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE_ECHO)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_ECHO)
    parser.add_argument("--epochs", type=int, default=EPOCHS_ECHO)
    parser.add_argument("--lr", type=float, default=LR_ECHO)
    args = parser.parse_args()

    start = time.time()

    ds = EchoDataset(delay=args.delay, seq_length=args.seq_length, size=args.size)
    dl = DataLoader(ds, batch_size=args.batch_size)

    model = GRUMemory(hidden_size=args.hidden_size)
    train_fixed_delay(model, dl, epochs=args.epochs, lr=args.lr)

    acc = test_model_fixed_delay(model, delay=args.delay)
    print(f"Accuracy (fixed delay={args.delay}): {acc:.4f}")

    assert_runtime(start)
    assert_accuracy_echo(acc)
    print("tests passed")

if __name__ == "__main__":
    main()

