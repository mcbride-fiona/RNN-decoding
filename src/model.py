import torch
import torch.nn as nn
from .config import N, DELAY_EMBED_DIM, POS_EMBED_DIM
from .utils import idx_to_onehot, chars_to_idx_tensor, idx_to_chars

class GRUMemory(nn.Module):
    """
    Fixed-delay GRU memory model.
    Input: one-hot (batch, seq, N+1)
    Output: logits over N+1 at each time step (batch, seq, N+1)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(N + 1, hidden_size)
        self.decoder = nn.Linear(hidden_size, N + 1)

    def forward(self, x, h0=None):
        # x: (B, T, N+1)
        B, T, _ = x.shape
        ht = torch.zeros(B, self.hidden_size, device=x.device) if h0 is None else h0
        logits = torch.zeros(B, T, N + 1, device=x.device)

        for t in range(T):
            ht = self.gru_cell(x[:, t, :], ht)
            logits[:, t, :] = self.decoder(ht)

        return logits

    @torch.no_grad()
    def test_run(self, s: str):
        """
        Accepts a lowercase 'a'..'z' string s.
        Returns the model's per-step predictions decoded to characters (space for 0).
        """
        x_idx = chars_to_idx_tensor(s).unsqueeze(0)   # (1, T)
        x_oh = idx_to_onehot(x_idx)                   # (1, T, N+1)
        logits = self.forward(x_oh)
        preds = torch.argmax(logits, dim=-1).squeeze(0)  # (T,)
        return idx_to_chars(preds)

class VariableDelayGRUMemory(nn.Module):
    """
    Variable-delay GRU with:
      - learned delay embeddings (init h0 + concat every step),
      - learned positional embeddings,
      - explicit per-step 'emit gate' g_t = 1[t >= delay] concatenated to input,
      - 2-layer GRU with dropout.
    """
    def __init__(self, hidden_size: int, max_delay: int,
                 delay_embed_dim: int = DELAY_EMBED_DIM,
                 pos_embed_dim: int = POS_EMBED_DIM,
                 max_seq_len: int = 512,
                 num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_delay = max_delay

        self.delay_emb = nn.Embedding(max_delay + 1, delay_embed_dim)
        self.pos_emb   = nn.Embedding(max_seq_len, pos_embed_dim)

        # +1 for emit-gate feature
        in_dim = (N + 1) + delay_embed_dim + pos_embed_dim + 1

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.decoder = nn.Linear(hidden_size, N + 1)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.init_from_delay = nn.Sequential(
            nn.Linear(delay_embed_dim, hidden_size),
            nn.Tanh()
        )

    def forward(self, x, delays, h0=None):
        """
        x: (B, T, N+1) one-hot; delays: (B,)
        """
        B, T, _ = x.shape
        device = x.device

        d = self.delay_emb(delays.long())                    # (B, Dd)
        p = self.pos_emb(torch.arange(T, device=device))     # (T, Dp)
        p = p.unsqueeze(0).expand(B, T, -1)                  # (B, T, Dp)
        d_time = d.unsqueeze(1).expand(B, T, -1)             # (B, T, Dd)

        # Emit gate g_t = 1[t >= delay]
        t_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B,T)
        g = (t_idx >= delays.unsqueeze(1)).float().unsqueeze(-1)          # (B,T,1)

        x_cat = torch.cat([x, d_time, p, g], dim=-1)         # (B, T, in_dim)

        h0_delay = self.init_from_delay(d).unsqueeze(0)      # (1, B, H)
        if h0 is not None:
            h0_delay = h0

        out, _ = self.gru(x_cat, h0_delay.repeat(self.gru.num_layers, 1, 1))  # (L=B,T,H)
        logits = self.decoder(out)                           # (B, T, N+1)
        return self.log_softmax(logits), None

    @torch.no_grad()
    def test_run(self, s: str, delay: int):
        x_idx = chars_to_idx_tensor(s).unsqueeze(0)
        x_oh  = idx_to_onehot(x_idx).to(self.decoder.weight.device)
        d = torch.tensor([delay], device=self.decoder.weight.device)
        log_probs, _ = self.forward(x_oh, d)
        preds = torch.argmax(log_probs, dim=-1).squeeze(0)
        return idx_to_chars(preds)