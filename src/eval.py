import time
import random
import string
import torch

from .config import DELAY, MAX_TOTAL_TRAIN_SECONDS, MIN_ACC_ECHO, MIN_ACC_VAR
from .utils import device

def test_model_fixed_delay(model, delay: int = DELAY, trials: int = 500) -> float:
    """
    Runs `trials` random strings; checks that output shifted by delay matches input.
    """
    total, correct = 0, 0
    for _ in range(trials):
        s = ''.join(random.choice(string.ascii_lowercase) for _ in range(random.randint(15, 25)))
        result = model.test_run(s)
        for c1, c2 in zip(s[:-delay], result[delay:]):
            correct += int(c1 == c2)
        total += len(s) - delay
    return correct / max(1, total)

def test_model_variable_delay(model, max_delay: int = 12, trials: int = 1000) -> float:
    """
    For each trial, sample a new delay in [0, max_delay] and evaluate shift accuracy.
    """
    total, correct = 0, 0
    for _ in range(trials):
        delay = random.randint(0, max_delay)
        s = ''.join(random.choice(string.ascii_lowercase) for _ in range(random.randint(15, 25)))
        result = model.test_run(s, delay)
        for c1, c2 in zip(s[:-delay], result[delay:]):
            correct += int(c1 == c2)
        total += len(s) - delay
    return correct / max(1, total)

def assert_runtime(start_time: float):
    dur = time.time() - start_time
    assert dur < MAX_TOTAL_TRAIN_SECONDS, f"execution took {dur:.2f}s (> {MAX_TOTAL_TRAIN_SECONDS}s limit)"

def assert_accuracy_echo(acc: float):
    assert acc > MIN_ACC_ECHO, f"accuracy too low: {acc:.4f} (need > {MIN_ACC_ECHO})"

def assert_accuracy_variable(acc: float):
    assert acc > MIN_ACC_VAR, f"accuracy too low: {acc:.4f} (need > {MIN_ACC_VAR})"

