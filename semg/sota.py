"""Reproductions of two published deep models for NinaPro DB1, each trained and
scored with its OWN paper's recipe (not the project-wide protocol in
protocols.py), so the numbers can be compared directly with the papers.

    hartwell : Hartwell, Kadirkamanathan & Anderson (2020), "A temporal-to-
               spatial deep convolutional neural network for classification of
               hand movements from multichannel electromyography data",
               arXiv:2007.10879.  Models: TtS (Table I), BaselineCNN (Table III).
    hu       : Hu, Wong, Wei, Du, Kankanhalli & Geng (2018), "A novel
               attention-based hybrid CNN-RNN architecture for sEMG-based
               gesture recognition", PLoS ONE 13(10):e0206049.
               Model: HuAttn (Table 1, raw-image1 input).

Every setting a paper does NOT report is listed in ASSUMPTIONS below and
printed at the start of each run, so the report can state them.
"""
import time

import numpy as np

from . import config as C
from . import deep
from .data import load_subject, make_windows

torch, nn = deep.torch, deep.nn

# ---------------------------------------------------------------------------
# Paper recipes
# ---------------------------------------------------------------------------
# Hartwell et al. Table V: 10 repetition splits (train reps, test reps).
# Split 1 is the NinaPro standard split used by Atzori et al. and Hu et al.
HARTWELL_SPLITS = [
    ([1, 3, 4, 6, 8, 9, 10], [2, 5, 7]),
    ([1, 2, 3, 5, 7, 9, 10], [4, 6, 8]),
    ([1, 2, 4, 6, 7, 8, 10], [3, 5, 9]),
    ([1, 2, 5, 6, 8, 9, 10], [3, 4, 7]),
    ([2, 3, 4, 6, 8, 9, 10], [1, 5, 7]),
    ([1, 2, 3, 4, 5, 7, 9], [6, 8, 10]),
    ([3, 5, 6, 7, 8, 9, 10], [1, 2, 4]),
    ([1, 2, 4, 5, 7, 8, 9], [3, 6, 10]),
    ([3, 4, 5, 6, 7, 8, 10], [1, 2, 9]),
    ([1, 2, 3, 4, 5, 6, 7], [8, 9, 10]),
]

RECIPES = {
    "hartwell": dict(
        models=["TtS", "BaselineCNN"],
        window=15, step=1,                     # 150 ms windows, 10 ms increment
        gestures=list(range(53)),              # 52 movements + rest, rest NOT down-sampled
        rest_mode="nearest",
        splits=HARTWELL_SPLITS,
        epochs=10, batch_size=32, optimizer="adam", lr=1e-3, weight_decay=0.0,
        class_weights=True,                    # Eq. 6
        # Table VI, inter-subject mean (SD) in %
        targets={"TtS": {"macro": (66.6, 5.1), "micro": (77.5, 4.5),
                         "macro_no_rep1": (69.3, 5.4), "micro_no_rep1": (78.0, 4.6)},
                 "BaselineCNN": {"macro": (65.0, 5.1), "micro": (77.1, 4.7)}},
    ),
    "hu": dict(
        models=["HuAttn"],
        window=20, step=1,                     # 200 ms windows
        gestures=list(range(1, 53)),           # 52 movements, no rest
        rest_mode="forward",
        splits=[(C.TRAIN_REPS, C.TEST_REPS)],  # "approximately 2/3 of the trials" as in Atzori/Geng
        subsegments=5,
        epochs=28, batch_size=1000, optimizer="sgd", lr=0.1, weight_decay=1e-4,
        lr_steps=[16, 24], loss_alpha=1.0, loss_beta=1.0,
        # Table 4, "Attention-based hybrid CNN-RNN with raw-image1", NinaProDB1
        targets={"HuAttn": {"window_acc": (84.8, None), "trial_acc": (96.5, None)}},
    ),
    "jiang": dict(
        models=["RIE"],
        window=30, step=5,                     # 300 ms windows, 50 ms step
        gestures=list(range(1, 53)),           # 52 movements, no rest
        rest_mode="forward",
        splits=[(C.TRAIN_REPS, C.TEST_REPS)],  # reps 2, 5, 7 held out (the paper's "validation set")
        denoise="sym4",
        epochs=150, batch_size=128, optimizer="adam", lr=1e-3, weight_decay=0.0, cosine=True,
        track_test=True,                       # test accuracy after every epoch: final vs best epoch
        # Table 3/4, RIE on NinaPro DB1 (52 gestures), mean over 27 subjects
        targets={"RIE": {"window_acc": (88.27, None), "best_epoch_acc": (88.27, None)}},
    ),
}

ASSUMPTIONS = {
    "hartwell": [
        "Batch size not reported: Keras default 32 used.",
        "Leaky-ReLU slope not reported: Keras default 0.3 used.",
        "Table I lists 'alpha = 0.001' in the Gaussian-noise row: read as noise SD 0.001 "
        "(on z-scored input).",
        "Windows that straddle a gesture/repetition boundary are discarded; the paper labels "
        "every window by its last sample and enforces a one-window gap between repetitions.",
    ],
    "hu": [
        "Only the raw-image1 input is reproduced; feature-signal-image1 (87.0%) is not "
        "specified well enough (feature list and 51-row layout missing).",
        "Optimizer, learning rate, schedule, batch size and epochs not reported: GengNet's "
        "recipe used (Geng et al. 2016: SGD, lr 0.1 divided by 10 after epochs 16 and 24, "
        "28 epochs, batch 1000, weight decay 1e-4). GengNet gives no momentum value: 0.9 assumed.",
        "Loss weights alpha, beta (Eq. 4) not reported: both 1; lambda = weight decay 1e-4.",
        "Number of subsegments for Table 4 not stated: 5 (non-overlapping, 4 frames each); "
        "Fig. 4 at 5 subsegments matches the Table 4 raw-image1 value of 84.8%.",
        "Window step not reported: 1 sample (10 ms) for training and test.",
        "Low-pass Butterworth filter order/cutoff not reported: off by default (--lowpass-hz).",
        "Activation in CNN layers not stated: ReLU. Input z-scored per channel on train data.",
        "Prediction uses the attention output; trial accuracy = majority vote over all test "
        "windows of one (gesture, repetition) trial.",
    ],
    "jiang": [
        "Symlet wavelet denoising: wavelet order, level and threshold not reported. sym4, "
        "level 4, soft universal threshold (sigma from the finest detail level), per channel.",
        "Z-score normalisation fitted on training repetitions only (paper: 'all sEMG data').",
        "Cosine annealing minimum learning rate not reported: 0.",
        "Dropout 0.5 'in the FC layer': placed between FC1 and FC2.",
        "The 1x1 shortcut conv gets BN + ReLU like every other conv ('all convolutional layers, "
        "except for the ECA block').",
        "The paper scores reps 2, 5, 7 as a 'validation set' and does not say whether its 88.27% "
        "is the final or the best epoch: both are reported (window_acc = final epoch).",
    ],
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _lowpass(sig, hz, order=1):
    from scipy.signal import butter, filtfilt
    b, a = butter(order, hz / (C.FS / 2))
    emg = sig["emg"].copy()
    for ex in np.unique(sig["exercise"]):
        m = sig["exercise"] == ex
        emg[m] = filtfilt(b, a, emg[m], axis=0)
    return {**sig, "emg": emg.astype(np.float32)}


def _wavelet_denoise(sig, wavelet="sym4", level=4):
    """Soft universal-threshold wavelet denoising per channel and exercise
    (Donoho & Johnstone), as a stand-in for Jiang et al.'s unspecified
    'Symlets wavelet decomposition'."""
    import pywt
    emg = sig["emg"].copy()
    for ex in np.unique(sig["exercise"]):
        m = sig["exercise"] == ex
        for c in range(emg.shape[1]):
            x = emg[m, c].astype(np.float64)
            coeffs = pywt.wavedec(x, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thr = sigma * np.sqrt(2 * np.log(len(x)))
            coeffs[1:] = [pywt.threshold(d, thr, mode="soft") for d in coeffs[1:]]
            emg[m, c] = pywt.waverec(coeffs, wavelet)[:len(x)]
    return {**sig, "emg": emg.astype(np.float32)}


def paper_windows(subject, recipe, lowpass_hz=None, data_dir=C.DATA_DIR):
    """All windows of one subject for a recipe (no rest down-sampling).
    Not cached: at a 1-sample step a subject is ~250 MB, and building the
    windows takes seconds next to minutes of training.
    Returns X (n, 10, W) float32, y (n,) global labels, rep (n,)."""
    sig = load_subject(subject, data_dir, rest_mode=recipe["rest_mode"])
    if lowpass_hz:
        sig = _lowpass(sig, lowpass_hz)
    if recipe.get("denoise"):
        sig = _wavelet_denoise(sig, recipe["denoise"])
    X, y, r = make_windows(sig, recipe["window"], recipe["step"])
    keep = np.isin(y, recipe["gestures"]) & (r > 0)
    return X[keep], y[keep], r[keep]


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------
if deep.TORCH_OK:

    class GaussianNoise(nn.Module):
        def __init__(self, sd):
            super().__init__()
            self.sd = sd

        def forward(self, x):
            return x + self.sd * torch.randn_like(x) if self.training and self.sd > 0 else x

    def _glorot(module):
        for m in module.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    class TtS(nn.Module):
        """Hartwell et al. Table I. Image = time (rows) x channels (cols)."""

        def __init__(self, n_channels, n_classes, window=15, slope=0.3, noise=1e-3):
            super().__init__()
            act = lambda: nn.LeakyReLU(slope)  # noqa: E731
            self.noise = GaussianNoise(noise)
            self.temporal = nn.Sequential(nn.Conv2d(1, 64, (3, 1), padding="same"), act())
            self.squeeze = nn.Sequential(nn.Conv2d(64, 32, 1), act())
            self.expand1 = nn.Sequential(nn.Conv2d(32, 64, 1), act())
            self.expand3 = nn.Sequential(nn.Conv2d(32, 64, (3, 1), padding="same"), act())
            self.spatial = nn.Sequential(nn.Conv2d(128, 32, (3, n_channels), padding="same"), act())
            self.head = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.5),
                nn.Linear(32 * window * n_channels, 128), act(), nn.Dropout(0.5),
                nn.Linear(128, n_classes))
            _glorot(self)

        def forward(self, x):                                  # (B, C, T)
            z = self.temporal(self.noise(x.transpose(1, 2).unsqueeze(1)))
            s = self.squeeze(z)
            z = torch.cat([self.expand1(s), self.expand3(s)], dim=1)
            return self.head(self.spatial(z))

    class BaselineCNN(nn.Module):
        """Hartwell et al. Table III (same size, no temporal constraint)."""

        def __init__(self, n_channels, n_classes, window=15, slope=0.3, noise=1e-3):
            super().__init__()
            act = lambda: nn.LeakyReLU(slope)  # noqa: E731
            self.net = nn.Sequential(
                GaussianNoise(noise),
                nn.Conv2d(1, 128, 3, padding="same"), act(),
                nn.Conv2d(128, 64, (5, 3), padding="same"), act(),
                nn.Conv2d(64, 32, (5, 3), padding="same"), act(),
                nn.Flatten(), nn.Dropout(0.5),
                nn.Linear(32 * window * n_channels, 128), act(), nn.Dropout(0.5),
                nn.Linear(128, n_classes))
            _glorot(self)

        def forward(self, x):
            return self.net(x.transpose(1, 2).unsqueeze(1))

    class LocallyConnected1x1(nn.Module):
        """1x1 convolution with separate (unshared) weights at every position."""

        def __init__(self, positions, cin, cout):
            super().__init__()
            bound = 1 / np.sqrt(cin)
            self.weight = nn.Parameter(torch.empty(positions, cin, cout).uniform_(-bound, bound))
            self.bias = nn.Parameter(torch.zeros(cout, positions))

        def forward(self, x):                                  # (B, cin, P)
            return torch.einsum("bcp,pcd->bdp", x, self.weight) + self.bias

    class HuAttn(nn.Module):
        """Hu et al. Table 1 with raw-image1 input.

        A window of L frames is cut into T subsegments of N = L/T frames; each
        becomes an N x 10 x 1 image (frames as colour planes). The CNN maps
        every image to a 128-d vector, an LSTM runs over the T vectors and
        attention pools the hidden states."""

        def __init__(self, n_channels, n_classes, window=20, subsegments=5, hidden=512):
            super().__init__()
            if window % subsegments:
                raise ValueError(f"window {window} not divisible by {subsegments} subsegments")
            self.T, self.N, self.C = subsegments, window // subsegments, n_channels
            self.cnn = nn.Sequential(
                nn.Conv2d(self.N, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Flatten(2),                                 # (B*T, 64, C)
                LocallyConnected1x1(n_channels, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
                LocallyConnected1x1(n_channels, 64, 64), nn.BatchNorm1d(64), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * n_channels, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
                nn.Linear(512, 128), nn.BatchNorm1d(128), nn.ReLU())
            self.lstm = nn.LSTM(128, hidden, batch_first=True)
            self.drop = nn.Dropout(0.5)
            self.att_W = nn.Linear(hidden, hidden, bias=False)
            self.att_w = nn.Linear(hidden, 1, bias=False)
            self.fc = nn.Linear(hidden, n_classes)

        def forward(self, x, return_steps=False):              # (B, C, L)
            B = x.shape[0]
            img = x.reshape(B, self.C, self.T, self.N).permute(0, 2, 3, 1)   # (B, T, N, C)
            f = self.cnn(img.reshape(B * self.T, self.N, self.C, 1)).reshape(B, self.T, -1)
            h, _ = self.lstm(f)                                # (B, T, H)
            a = torch.softmax(self.att_w(torch.tanh(self.att_W(h))), dim=1)  # (B, T, 1)
            out = self.fc(self.drop((a * h).sum(1)))
            if return_steps:
                return out, self.fc(self.drop(h))              # (B, G), (B, T, G)
            return out

    def _cbr(cin, cout, k):
        """Conv (stride 1, 'same' padding) + BatchNorm + ReLU."""
        pad = (k[0] // 2, k[1] // 2) if isinstance(k, tuple) else k // 2
        return nn.Sequential(nn.Conv2d(cin, cout, k, padding=pad), nn.BatchNorm2d(cout), nn.ReLU())

    class ECA(nn.Module):
        """Efficient channel attention: GAP -> 1D conv over channels (k=15) -> sigmoid."""

        def __init__(self, k=15):
            super().__init__()
            self.conv = nn.Conv1d(1, 1, k, padding=k // 2, bias=False)

        def forward(self, x):                                  # (B, C, H, W)
            w = self.conv(x.mean((2, 3)).unsqueeze(1))         # (B, 1, C)
            return x * torch.sigmoid(w).squeeze(1)[:, :, None, None]

    class INECA(nn.Module):
        """Jiang et al. Fig. 1: four Inception paths + 1x1 residual shortcut, then ECA."""

        def __init__(self, cin, ch):
            super().__init__()
            c1, c2, c3, c4, c5, c6 = ch
            self.p1 = _cbr(cin, c1, 1)
            self.p2 = nn.Sequential(_cbr(cin, c2, 1), _cbr(c2, c3, 3))
            self.p3 = nn.Sequential(_cbr(cin, c4, 1), _cbr(c4, c5, (5, 1)), _cbr(c5, c5, (1, 5)))
            self.p4 = nn.Sequential(nn.AvgPool2d(3, stride=1, padding=1), _cbr(cin, c6, 1))
            self.res = _cbr(cin, c1 + c3 + c5 + c6, 1)
            self.eca = ECA(15)

        def forward(self, x):
            out = torch.cat([self.p1(x), self.p2(x), self.p3(x), self.p4(x)], dim=1)
            return self.eca(out + self.res(x))

    class RIE(nn.Module):
        """Jiang et al. 2024 (Table 1): 4 IN-ECA blocks -> average pool to 2x2 -> FC 128 -> FC K.
        Input image = time (30 rows) x channels (10 columns)."""

        BLOCKS = [(16, 12, 32, 4, 8, 8), (32, 24, 64, 8, 16, 16),
                  (64, 48, 128, 16, 32, 32), (128, 96, 256, 32, 64, 64)]

        def __init__(self, n_channels, n_classes):
            super().__init__()
            layers, cin = [], 1
            for ch in self.BLOCKS:
                layers.append(INECA(cin, ch))
                cin = ch[0] + ch[2] + ch[4] + ch[5]
            self.features = nn.Sequential(*layers)
            self.pool = nn.AdaptiveAvgPool2d((2, 2))           # "from 30 x 10 to 2 x 2"
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(cin * 4, 128), nn.ReLU(),
                                      nn.Dropout(0.5), nn.Linear(128, n_classes))

        def forward(self, x):                                  # (B, C, T)
            return self.head(self.pool(self.features(x.transpose(1, 2).unsqueeze(1))))

    ARCHS = {"TtS": TtS, "BaselineCNN": BaselineCNN, "HuAttn": HuAttn, "RIE": RIE}


def make_net(name, n_classes, recipe):
    deep._require_torch()
    if name == "HuAttn":
        return HuAttn(C.N_CHANNELS, n_classes, recipe["window"], recipe["subsegments"])
    if name == "RIE":
        return RIE(C.N_CHANNELS, n_classes)
    return ARCHS[name](C.N_CHANNELS, n_classes, recipe["window"])


def n_params(net):
    return sum(p.numel() for p in net.parameters())


# ---------------------------------------------------------------------------
# Training (fixed number of epochs, no early stopping, as in both papers)
# ---------------------------------------------------------------------------
def hartwell_class_weights(y, n_classes):
    """Eq. 6: gamma_j = 1 + log2(n_max / n_j)."""
    n = np.bincount(y, minlength=n_classes).astype(np.float64)
    return (1 + np.log2(n.max() / np.maximum(n, 1))).astype(np.float32)


def train(name, Xtr, ytr, n_classes, recipe, seed=C.SEED, verbose=True, on_epoch=None):
    """on_epoch: optional callable(net) run after every epoch (e.g. test accuracy);
    it only observes, training never sees its result."""
    deep.set_seed(seed)
    device = deep.get_device()
    net = make_net(name, n_classes, recipe).to(device)
    if recipe["optimizer"] == "adam":
        opt = torch.optim.Adam(net.parameters(), lr=recipe["lr"], betas=(0.9, 0.999),
                               weight_decay=recipe["weight_decay"])
    else:
        opt = torch.optim.SGD(net.parameters(), lr=recipe["lr"], momentum=0.9,
                              weight_decay=recipe["weight_decay"])
    if recipe.get("lr_steps"):
        sched = torch.optim.lr_scheduler.MultiStepLR(opt, recipe["lr_steps"], 0.1)
    elif recipe.get("cosine"):
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=recipe["epochs"], eta_min=0.0)
    else:
        sched = None

    is_hu = name == "HuAttn"
    cw = (torch.tensor(hartwell_class_weights(ytr, n_classes), device=device)
          if recipe.get("class_weights") else torch.ones(n_classes, device=device))
    ce = nn.CrossEntropyLoss(reduction="none")

    def loss_fn(xb, yb):
        if is_hu:  # Eq. 4-10: attention loss / T + mean per-step (target replication) loss
            out, steps = net(xb, return_steps=True)
            T = steps.shape[1]
            l_att = ce(out, yb).mean() / T
            l_tgt = ce(steps.reshape(-1, steps.shape[-1]), yb.repeat_interleave(T)).mean()
            return recipe["loss_alpha"] * l_att + recipe["loss_beta"] * l_tgt
        # Keras class_weight style (Hartwell): per-sample loss scaled by its class weight,
        # averaged over the batch; all weights 1 = plain cross-entropy
        return (ce(net(xb), yb) * cw[yb]).mean()

    # whole training set lives on the device: with batch 32 the per-batch
    # host->GPU copies would otherwise dominate (~200 MB, fits easily)
    X_all = torch.from_numpy(np.ascontiguousarray(Xtr, dtype=np.float32)).to(device)
    y_all = torch.from_numpy(ytr.astype(np.int64)).to(device)
    rng = np.random.default_rng(seed)
    bs = recipe["batch_size"]
    for epoch in range(1, recipe["epochs"] + 1):
        t0 = time.time()
        net.train()
        order = torch.from_numpy(rng.permutation(len(ytr))).to(device)
        total = 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            if len(idx) < 2 and is_hu:  # BatchNorm needs > 1 sample
                continue
            xb, yb = X_all[idx], y_all[idx]
            opt.zero_grad()
            loss = loss_fn(xb, yb)
            loss.backward()
            opt.step()
            total += loss.detach() * len(idx)  # no .item(): avoids a GPU sync every batch
        if sched:
            sched.step()
        extra = ""
        if on_epoch is not None:
            extra = f"  {on_epoch(net)}"
        if verbose:
            print(f"      epoch {epoch:>2}/{recipe['epochs']}  loss {float(total) / len(ytr):.4f}"
                  f"{extra}  ({time.time() - t0:.0f}s)", flush=True)
    return net


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def per_class_counts(y, pred, n_classes):
    """True positives and support per class (for Forman pooling, Eq. 14-15)."""
    tp = np.bincount(y[pred == y], minlength=n_classes)
    n = np.bincount(y, minlength=n_classes)
    return tp, n


def trial_vote(y, rep, pred):
    """Majority vote over all windows of one (gesture, repetition) trial."""
    correct = total = 0
    for g, r in set(zip(y.tolist(), rep.tolist())):
        m = (y == g) & (rep == r)
        correct += int(np.bincount(pred[m]).argmax() == g)
        total += 1
    return correct, total


def fold_result(y, rep, pred, n_classes):
    """Everything needed to aggregate a fold later (small arrays only)."""
    tp, n = per_class_counts(y, pred, n_classes)
    no1 = rep != 1
    tp1, n1 = per_class_counts(y[no1], pred[no1], n_classes)
    tc, tt = trial_vote(y, rep, pred)
    return dict(tp=tp, n=n, tp_no_rep1=tp1, n_no_rep1=n1, trial_correct=tc, trial_total=tt)


def subject_metrics(folds):
    """Pool a subject's folds as Hartwell et al. Eq. 14-15 (sum TP and support
    over folds, then micro = pooled accuracy, macro = mean per-class recall)."""
    out = {}
    for suffix in ("", "_no_rep1"):
        tp = sum(f["tp" + suffix] for f in folds)
        n = sum(f["n" + suffix] for f in folds)
        present = n > 0
        out["micro" + suffix] = 100 * tp.sum() / max(n.sum(), 1)
        out["macro" + suffix] = 100 * np.mean(tp[present] / n[present])
    out["window_acc"] = out["micro"]
    out["trial_acc"] = 100 * sum(f["trial_correct"] for f in folds) / max(sum(f["trial_total"] for f in folds), 1)
    curves = [f["epoch_acc"] for f in folds if "epoch_acc" in f and len(f["epoch_acc"])]
    if curves:  # test accuracy after every epoch: the best one is what test-set model selection would report
        out["best_epoch_acc"] = 100 * float(np.mean([c.max() for c in curves]))
    return out
