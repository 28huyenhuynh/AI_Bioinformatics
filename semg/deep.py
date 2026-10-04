"""Deep learning models (PyTorch): 1D-CNN, LSTM, CNN-LSTM and a 2D "image" CNN.

Input to every model: windows of shape (batch, channels=10, time=W).

PyTorch is imported lazily so the classical part of the project works even
when torch is not installed.
"""
import copy
import time

import numpy as np

from . import config as C

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:  # pragma: no cover
    torch, nn = None, None
    TORCH_OK = False


def _require_torch():
    if not TORCH_OK:
        raise ImportError("PyTorch is not installed. Install it with: pip install torch")


# ---------------------------------------------------------------------------
# Normalisation (fit on TRAIN only)
# ---------------------------------------------------------------------------
class ChannelStandardizer:
    """Per-channel z-score. Optional log transform first, which makes the
    skewed DB1 envelope closer to Gaussian."""

    def __init__(self, log=False, eps=1e-3):
        self.log, self.eps = log, eps

    def _pre(self, X):
        return np.log(X + self.eps) if self.log else X

    def fit(self, X):
        Z = self._pre(X)
        self.mean_ = Z.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std_ = (Z.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
        return self

    def transform(self, X):
        return ((self._pre(X) - self.mean_) / self.std_).astype(np.float32)


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------
if TORCH_OK:

    class CNN1D(nn.Module):
        """Simple 1D-CNN: convolutions along time, channels as input planes."""

        def __init__(self, n_channels, n_classes, width=64, dropout=0.3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_channels, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
                nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(width, 2 * width, 3, padding=1), nn.BatchNorm1d(2 * width), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(2 * width, n_classes))

        def forward(self, x):
            return self.head(self.features(x))

    class LSTMNet(nn.Module):
        """2-layer LSTM over time; last hidden state -> classifier."""

        def __init__(self, n_channels, n_classes, hidden=64, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(n_channels, hidden, num_layers=2, batch_first=True, dropout=dropout)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

        def forward(self, x):                      # x: (B, C, T)
            out, _ = self.lstm(x.transpose(1, 2))  # (B, T, H)
            return self.head(out[:, -1])

    class CNNLSTM(nn.Module):
        """Conv block extracts local patterns, LSTM models their sequence."""

        def __init__(self, n_channels, n_classes, width=64, hidden=64, dropout=0.3):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_channels, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
                nn.Conv1d(width, width, 3, padding=1), nn.BatchNorm1d(width), nn.ReLU(),
            )
            self.lstm = nn.LSTM(width, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_classes))

        def forward(self, x):
            h = self.conv(x).transpose(1, 2)
            out, _ = self.lstm(h)
            return self.head(out[:, -1])

    class CNN2D(nn.Module):
        """Treats a window as a 1 x time x channel image, in the spirit of
        Atzori et al. (2016). Starting point for that reproduction: compare
        layer sizes with the paper's architecture figure and adjust."""

        def __init__(self, n_channels, n_classes, dropout=0.3):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(64, n_classes))

        def forward(self, x):                      # (B, C, T) -> (B, 1, T, C)
            return self.head(self.features(x.transpose(1, 2).unsqueeze(1)))

    ARCHS = {"CNN1D": CNN1D, "LSTM": LSTMNet, "CNNLSTM": CNNLSTM, "CNN2D": CNN2D}


def make_net(name, n_channels, n_classes):
    _require_torch()
    if name not in ARCHS:
        raise ValueError(f"Unknown deep model {name}; choose from {list(ARCHS)}")
    return ARCHS[name](n_channels, n_classes)


def get_device():
    _require_torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    _require_torch()
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_net(name, Xtr, ytr, Xval, yval, n_classes, epochs=40, batch_size=256, lr=1e-3,
              weight_decay=1e-4, patience=8, seed=C.SEED, augment=None, verbose=True):
    """Train with Adam + class-weighted cross-entropy and early stopping on
    validation loss. X arrays are already normalised, labels are 0..K-1.

    augment: optional callable(X_batch_numpy, rng) -> X_batch_numpy applied to
    each training batch (see perturb.augment_batch).
    """
    _require_torch()
    set_seed(seed)
    device = get_device()
    net = make_net(name, Xtr.shape[1], n_classes).to(device)

    counts = np.bincount(ytr, minlength=n_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1) / n_classes
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)

    Xval_t = torch.from_numpy(Xval).to(device)
    yval_t = torch.from_numpy(yval.astype(np.int64)).to(device)
    rng = np.random.default_rng(seed)
    best, best_state, bad = np.inf, None, 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        order = rng.permutation(len(ytr))
        total = 0.0
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            xb = Xtr[idx]
            if augment is not None:
                xb = augment(xb, rng)
            xb = torch.from_numpy(np.ascontiguousarray(xb, dtype=np.float32)).to(device)
            yb = torch.from_numpy(ytr[idx].astype(np.int64)).to(device)
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)

        val_loss = _eval_loss(net, Xval_t, yval_t, loss_fn, batch_size)
        sched.step(val_loss)
        if verbose:
            print(f"    epoch {epoch:>3}  train {total / len(ytr):.4f}  val {val_loss:.4f}"
                  f"  ({time.time() - t0:.1f}s)", flush=True)
        if val_loss < best - 1e-4:
            best, best_state, bad = val_loss, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    return net


def _eval_loss(net, X, y, loss_fn, batch_size):
    net.eval()
    total = 0.0
    with torch.no_grad():
        for i in range(0, len(y), batch_size * 4):
            total += loss_fn(net(X[i:i + batch_size * 4]), y[i:i + batch_size * 4]).item() * len(y[i:i + batch_size * 4])
    return total / max(len(y), 1)


def predict_proba(net, X, batch_size=1024):
    _require_torch()
    device = next(net.parameters()).device
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(np.ascontiguousarray(X[i:i + batch_size], dtype=np.float32)).to(device)
            out.append(torch.softmax(net(xb), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, 0))
