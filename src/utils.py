import torch
from .config import N

def idx_to_onehot(x: torch.Tensor, k: int = N + 1) -> torch.Tensor:
    """
    Convert an integer tensor (0..k-1) to one-hot encoding.
    Input shape: (...), dtype long/int64
    Output shape: (..., k)
    """
    # torch.eye is simpler and faster; keep device consistent
    eye = torch.eye(k, device=x.device, dtype=torch.float32)
    flat = x.view(-1).long()
    oh = eye.index_select(0, flat)
    return oh.view(*x.shape, k)

def chars_to_idx_tensor(s: str) -> torch.Tensor:
    """
    'a'..'z' -> 1..26 ; space (blank) -> 0
    """
    return torch.tensor([ord(c) - ord('a') + 1 for c in s], dtype=torch.int64)

def idx_to_chars(indices: torch.Tensor) -> str:
    """
    Inverse of chars_to_idx_tensor; 0 -> ' ', 1..26 -> 'a'..'z'
    """
    out = []
    for p in indices.view(-1):
        if int(p) == 0:
            out.append(' ')
        else:
            out.append(chr(int(p) + ord('a') - 1))
    return ''.join(out)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
