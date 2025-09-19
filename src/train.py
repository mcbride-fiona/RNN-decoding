import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm

from .config import (
    N, LR_VAR, LR_ECHO, EPOCHS_VAR, EPOCHS_ECHO, STEP_SIZE_VAR, GAMMA_VAR, MAX_TOTAL_TRAIN_SECONDS
)

from .utils import idx_to_onehot, device

def train_fixed_delay(model, dataloader, epochs: int = EPOCHS_ECHO, lr: float = LR_ECHO):
    dev = device()
    model.to(dev)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    start = time.time()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for seq, result in tqdm(dataloader, desc=f"[Echo] Epoch {epoch+1}/{epochs}"):
            seq = idx_to_onehot(seq.to(dev))        # (B, T, N+1)
            result = result.long().to(dev)          # (B, T)

            optimizer.zero_grad()
            logits = model(seq)                     # (B, T, N+1)
            loss = criterion(logits.view(-1, N + 1), result.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

        # Optional training time check (like original)
        if time.time() - start > MAX_TOTAL_TRAIN_SECONDS:
            print("Stopping early: exceeded max training time.")
            break

def train_variable_delay(model, dataloader, epochs: int = EPOCHS_VAR, lr: float = LR_VAR):
    dev = device()
    model.to(dev)
    model.train()

    # AdamW tends to help here
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE_VAR, gamma=GAMMA_VAR)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    criterion = nn.NLLLoss(reduction="none")

    start = time.time()
    for epoch in range(epochs):
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        for seq, delay, target in tqdm(dataloader, desc=f"[VarEcho] Epoch {epoch+1}/{epochs}"):
            seq    = idx_to_onehot(seq.to(dev))      # (B, T, N+1)
            delay  = delay.to(dev).long()            # (B,)
            target = target.to(dev).long()           # (B, T)

            B, T, _ = seq.shape
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=torch.cuda.is_available()):
                log_probs, _ = model(seq, delay)     # (B, T, N+1)

                # ---- mask: only positions t >= delay_i for each sample i ----
                t_idx = torch.arange(T, device=dev).unsqueeze(0).expand(B, T)   # (B, T)
                mask = (t_idx >= delay.unsqueeze(1)).float()                    # (B, T)

                lp_flat = log_probs.view(B * T, N + 1)
                tgt_flat = target.view(B * T)
                loss_per_pos = criterion(lp_flat, tgt_flat).view(B, T)          # (B, T)

                masked_loss = (loss_per_pos * mask).sum()
                denom = mask.sum().clamp_min(1.0)  # avoid div by zero if delay==T
                loss = masked_loss / denom

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += masked_loss.detach().item()
            total_tokens += denom.detach().item()
            num_batches += 1

        scheduler.step()
        avg = total_loss / max(1.0, total_tokens)
        print(f"Epoch {epoch+1}: masked loss/token={avg:.4f}")

        if time.time() - start > MAX_TOTAL_TRAIN_SECONDS:
            print("Stopping early: exceeded max training time.")
            break