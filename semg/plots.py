"""Figures for the report (matplotlib only, saved as PNG at 200 dpi)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import config as C  # noqa: E402

PALETTE = ["#2a6fdb", "#e3793b", "#2ca58d", "#c44e7a", "#8a6bd1", "#7f8c8d", "#d4a017", "#4bb3e0", "#a0522d"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def per_subject_bars(df, path, metric="macro_f1", title=None):
    models = list(df.groupby("model")[metric].mean().sort_values(ascending=False).index)
    subjects = sorted(df["subject"].unique())
    w = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(8, len(subjects) * 0.55), 4.2))
    x = np.arange(len(subjects))
    for i, m in enumerate(models):
        vals = df[df.model == m].set_index("subject").reindex(subjects)[metric]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=f"{m} (mean {vals.mean():.2f})",
               color=PALETTE[i % len(PALETTE)])
    ax.set_xticks(x, [f"S{s}" for s in subjects], rotation=90)
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_ylim(0, 1)
    ax.set_title(title or f"Per-subject {metric.replace('_', ' ')}")
    ax.legend(ncol=min(len(models), 4), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def model_boxplot(df, path, metric="macro_f1", title=None):
    order = list(df.groupby("model")[metric].median().sort_values(ascending=False).index)
    data = [df[df.model == m][metric].to_numpy() for m in order]
    fig, ax = plt.subplots(figsize=(1.3 * len(order) + 2, 4))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(order) + 1), order)
    for i, d in enumerate(data):
        ax.scatter(np.random.default_rng(i).normal(i + 1, 0.05, len(d)), d, s=10, alpha=0.6, color=PALETTE[i % len(PALETTE)])
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(title or f"{metric.replace('_', ' ')} across subjects")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, path)


def confusion_plot(cm, classes, path, title="Confusion matrix (row-normalised)"):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    K = len(classes)
    size = max(5, 0.32 * K + 3)
    fig, ax = plt.subplots(figsize=(size, size * 0.9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    names = [C.gesture_name(c) if K <= 12 else str(c) for c in classes]
    ax.set_xticks(range(K), names, rotation=90, fontsize=7 if K > 12 else 8)
    ax.set_yticks(range(K), names, fontsize=7 if K > 12 else 8)
    if K <= 12:
        for i in range(K):
            for j in range(K):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def degradation_curves(df, path, metric="macro_f1"):
    kinds = [k for k in ["noise", "fatigue", "rotation", "dropout"] if k in df.condition.unique()]
    xlabels = {"noise": "SNR (dB)  ← cleaner | noisier →", "fatigue": "Amplitude gain",
               "rotation": "Ring rotation (electrode positions)", "dropout": "Channels lost"}
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.2 * len(kinds), 3.6), squeeze=False)
    models = sorted(df.model.unique())
    for ax, kind in zip(axes[0], kinds):
        sub = df[df.condition == kind].copy()
        for i, m in enumerate(models):
            g = sub[sub.model == m].groupby("level")[metric]
            mean, sem = g.mean(), g.sem()
            levels = list(mean.index)
            if kind == "noise":  # clean first, then decreasing SNR
                levels = sorted(levels, key=lambda v: -np.inf if v in ("", "clean") else -float(v))
            else:
                levels = sorted(levels, key=float)
            xs = np.arange(len(levels))
            ax.errorbar(xs, mean[levels], yerr=sem[levels], marker="o", ms=4, capsize=2,
                        label=m, color=PALETTE[i % len(PALETTE)])
            ax.set_xticks(xs, ["clean" if v in ("", "clean") else v for v in levels])
        ax.set_xlabel(xlabels[kind])
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_title(kind.capitalize())
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    _save(fig, path)


def heatmap(M, row_labels, col_labels, path, title, cmap="viridis", fmt="{:.2f}", cbar_label=""):
    fig, ax = plt.subplots(figsize=(0.65 * len(col_labels) + 3.5, 0.45 * len(row_labels) + 1.8))
    im = ax.imshow(M, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(col_labels)), col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)), row_labels, fontsize=8)
    if M.size <= 400:
        lo, hi = np.nanmin(M), np.nanmax(M)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color="white" if (v - lo) < 0.55 * (hi - lo) else "black")
    ax.set_title(title)
    cb = fig.colorbar(im, ax=ax, fraction=0.046)
    cb.set_label(cbar_label)
    _save(fig, path)


def line_plot(x, ys: dict, path, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, (name, y) in enumerate(ys.items()):
        ax.plot(x, y, marker="o", label=name, color=PALETTE[i % len(PALETTE)])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, path)


def scatter(x, y, labels, path, xlabel, ylabel, title, annotate=True):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.scatter(x, y, color=PALETTE[0])
    if annotate:
        for xi, yi, l in zip(x, y, labels):
            ax.annotate(str(l), (xi, yi), fontsize=7, xytext=(3, 3), textcoords="offset points")
    if len(x) > 2:
        k, b = np.polyfit(x, y, 1)
        xs = np.linspace(np.min(x), np.max(x), 50)
        ax.plot(xs, k * xs + b, color=PALETTE[1], lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    _save(fig, path)
