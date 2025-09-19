import random
import torch
from torch.utils.data import IterableDataset
from .config import N

class EchoDataset(IterableDataset):
    """
    Fixed-delay echo: target is input shifted right by `delay`, left-padded with blanks (0).
    Yields: (seq, result) as int64 tensors, where tokens are 0..N (0 is blank, 1..26 letters).
    """
    def __init__(self, delay: int = 3, seq_length: int = 15, size: int = 1000):
        self.delay = delay
        self.seq_length = seq_length
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        for _ in range(self.size):
            seq = torch.tensor(
                [random.choice(range(1, N + 1)) for _ in range(self.seq_length)],
                dtype=torch.int64
            )
            result = torch.cat(
                (torch.zeros(self.delay, dtype=torch.int64),
                 seq[: self.seq_length - self.delay])
            )
            yield seq, result

class VariableDelayEchoDatasetBalanced(IterableDataset):
    """
    Ensures uniform coverage of delays 0..max_delay each 'cycle'.
    Useful when training with small epochs/batches.
    """
    def __init__(self, max_delay: int = 12, seq_length: int = 20, size: int = 60000):
        self.max_delay = max_delay
        self.seq_length = seq_length
        self.size = size

    def __len__(self):
        return self.size

    def __iter__(self):
        # Build a pool with uniform delays
        delays = list(range(self.max_delay + 1))
        idx = 0
        for _ in range(self.size):
            if idx == 0:
                random.shuffle(delays)
            delay = delays[idx]
            idx = (idx + 1) % len(delays)

            seq = torch.tensor(
                [random.randint(1, N) for _ in range(self.seq_length)],
                dtype=torch.int64
            )
            result = torch.cat(
                (torch.zeros(delay, dtype=torch.int64),
                 seq[: self.seq_length - delay])
            )
            yield seq, torch.tensor(delay, dtype=torch.int64), result