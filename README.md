# RNN Decoding — Echo Tasks with GRU

This repository implements **fixed-delay** and **variable-delay** **echo tasks** using GRU-based recurrent neural networks in PyTorch.  
The model receives a sequence of characters (`a`–`z`) and must output the same sequence shifted by a delay.  
This is a classic diagnostic task for testing the **memory capacity** of RNNs.

---

## Tasks

### Fixed Delay Echo
- Every sequence uses the same fixed delay (e.g. 2).
- The target sequence is the input shifted right by `delay` positions, left-padded with blank tokens (0).

**Example:**
| Input  | a | b | c | d | e |
|--------|---|---|---|---|---|
| Target | ␣ | ␣ | a | b | c |

---

### Variable Delay Echo
- Each sequence has its own random delay (0 … max_delay).
- Harder: the model must infer *when* to start echoing, not just *what* to echo.
- Includes:
  - Learned token, delay, and positional embeddings
  - Optional emit-gate feature (`1[t ≥ delay]`)
  - Masked loss (only positions `t ≥ delay` are supervised)
  - Balanced delay sampling for stable learning

**Example (delay = 3):**
| Input  | a | b | c | d | e |
|--------|---|---|---|---|---|
| Target | ␣ | ␣ | ␣ | a | b |

---
```bash
Repository Structure
├── src/   
│   ├── config.py              # Hyperparameters
│   ├── utils.py               # One-hot / token conversions
│   ├── datasets.py            # Fixed and variable-delay datasets
│   ├── model.py               # GRU models (fixed + variable)
│   ├── train.py               # Training loops (masked loss for variable)
│   ├── eval.py                # Accuracy evaluation
│   ├── main_echo.py           # CLI for fixed-delay task
│   └── main_variable.py       # CLI for variable-delay task
└── requirements.txt           # dependencies
```
---

## Usage

### Fixed Delay Task
python -m src.main_echo \
  --delay 4 --seq-length 15 --size 200000 \
  --hidden-size 128 --batch-size 64 --epochs 5 --lr 1e-3

### Variable Delay Task
python -m src.main_variable \
  --max-delay 12 --seq-length 32 --size 80000 \
  --hidden-size 192 --batch-size 128 --epochs 16 --lr 3e-3

