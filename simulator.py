"""STEP 6 - Interactive prosthetic-control simulator (replay of real DB1 data).

Trains a model on every subject EXCEPT the one you replay (so it behaves like
a prosthesis fitted to a new user), then streams that subject's recording
window by window, as a real controller would every 50 ms:

    left       : muscle activation on the forearm ring (ch1-8) + FDS / EDS
    top right  : scrolling sEMG envelopes
    mid right  : class probabilities
    bottom     : true vs predicted gesture, with majority-vote smoothing

Examples
    python simulator.py --subject 5 --model RF            # interactive window
    python simulator.py --subject 5 --model CNN1D         # needs PyTorch
    python simulator.py --subject 5 --save results/simulator/demo.gif --frames 300
"""
import argparse
import os
from collections import Counter, deque

import joblib
import numpy as np

from semg import config as C
from semg.data import build_dataset, list_subjects, load_subject, resolve_gestures
from semg.protocols import FittedModel, encoder


def build_stream(subject, gestures, data_dir, max_reps=3):
    """Continuous segment of the subject's recording that contains the chosen
    gestures (first `max_reps` repetitions of each, with the rest in between)."""
    sig = load_subject(subject, data_dir)
    lab = sig["label"]
    # label of the most recent movement (forward fill over rest periods)
    last = np.where(lab > 0, np.arange(len(lab)), 0)
    np.maximum.accumulate(last, out=last)
    prev_move = lab[last]
    moving = (lab > 0) & np.isin(lab, gestures)
    rest_after_kept = (lab == 0) & np.isin(prev_move, gestures) & (0 in gestures)
    keep = (moving | rest_after_kept) & (sig["rep"] <= max_reps)
    idx = np.where(keep)[0]
    return sig["emg"][idx], sig["label"][idx]


def get_model(args, subjects, gestures):
    path = os.path.join(C.RESULTS_DIR, "simulator", f"{args.model}_without_s{args.subject}_{args.gestures}.joblib")
    if os.path.exists(path) and not args.retrain:
        print(f"Loading cached model {path}")
        return joblib.load(path)
    train_subjects = [s for s in subjects if s != args.subject]
    print(f"Training {args.model} on {len(train_subjects)} subjects (S{args.subject} held out) ...")
    data = build_dataset(train_subjects, gestures, verbose=False, data_dir=args.data_dir)
    _, lut = encoder(gestures)
    val = np.isin(data["subject"], train_subjects[:2])
    fm = FittedModel(args.model, {"epochs": args.epochs}).fit(data["X"], lut[data["y"]], len(gestures), val_mask=val)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(fm, path)
    return fm


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", type=int, default=1, help="subject to replay (unseen by the model)")
    p.add_argument("--model", default="RF", help="RF, LDA, kNN5, CNN1D, LSTM ...")
    p.add_argument("--gestures", default="subset8")
    p.add_argument("--smooth", type=int, default=5, help="majority vote over the last N decisions (1 = off)")
    p.add_argument("--step", type=int, default=C.STEP, help="decision every N samples (5 = 50 ms)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--save", default=None, help="save an animation (.gif) instead of opening a window")
    p.add_argument("--frames", type=int, default=400, help="frames to render when saving")
    p.add_argument("--data-dir", default=C.DATA_DIR)
    args = p.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.widgets import Button, Slider

    subjects = list_subjects(args.data_dir)
    gestures = resolve_gestures(args.gestures)
    classes, lut = encoder(gestures)
    names = [C.gesture_name(g) for g in classes]
    model = get_model(args, subjects, gestures)
    emg, labels = build_stream(args.subject, gestures, args.data_dir)
    W = C.WINDOW
    starts = np.arange(0, len(emg) - W, args.step)
    vmax = np.percentile(emg, 99.5, axis=0)

    # ---- figure layout ----------------------------------------------------
    fig = plt.figure(figsize=(13, 7.2))
    fig.suptitle(f"sEMG prosthetic control simulator  -  {args.model} replaying unseen subject S{args.subject}",
                 fontsize=12, fontweight="bold")
    ax_ring = fig.add_axes([0.03, 0.34, 0.30, 0.52], projection="polar")
    ax_fe = fig.add_axes([0.09, 0.10, 0.20, 0.11])
    ax_sig = fig.add_axes([0.40, 0.62, 0.57, 0.27])
    ax_prob = fig.add_axes([0.52, 0.27, 0.45, 0.28])
    ax_txt = fig.add_axes([0.40, 0.02, 0.57, 0.17]); ax_txt.axis("off")

    theta = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    ring_bars = ax_ring.bar(theta, np.zeros(8), width=2 * np.pi / 8 * 0.85, color="#2a6fdb", alpha=0.85)
    ax_ring.set_ylim(0, 1.05)
    ax_ring.set_xticks(theta, [f"ch{i + 1}" for i in range(8)])
    ax_ring.set_yticklabels([])
    ax_ring.set_title("Forearm ring (ch1-8)", fontsize=10)
    fe_bars = ax_fe.barh(["ch9 FDS", "ch10 EDS"], [0, 0], color=["#e3793b", "#2ca58d"])
    ax_fe.set_xlim(0, 1.05)
    ax_fe.set_title("Finger flexor / extensor", fontsize=9)

    hist = 300  # samples shown (3 s)
    sig_lines = [ax_sig.plot([], [], lw=0.9, label=f"ch{c + 1}")[0] for c in range(C.N_CHANNELS)]
    ax_sig.set_xlim(0, hist); ax_sig.set_ylim(0, 1.05)
    ax_sig.set_title("Normalised sEMG envelopes (last 3 s)", fontsize=10)
    ax_sig.legend(ncol=10, fontsize=6, loc="upper left", frameon=False)
    ax_sig.set_xticks([])

    prob_bars = ax_prob.barh(names, np.zeros(len(names)), color="#7f8c8d")
    ax_prob.set_xlim(0, 1); ax_prob.invert_yaxis()
    ax_prob.set_title("Classifier output", fontsize=10)
    ax_prob.tick_params(axis="y", labelsize=8)

    txt = ax_txt.text(0.0, 0.62, "", fontsize=15, fontweight="bold", transform=ax_txt.transAxes)
    sub = ax_txt.text(0.0, 0.15, "", fontsize=10, transform=ax_txt.transAxes)

    state = {"i": 0, "playing": True, "speed": 1, "votes": deque(maxlen=max(1, args.smooth)),
             "n": 0, "ok": 0}

    def scores_for(window):
        s = np.asarray(model.scores(window[None]))[0]
        if s.min() < 0 or s.max() > 1:   # decision function -> softmax for display
            s = np.exp(s - s.max()); s /= s.sum()
        return s

    def update(_):
        if not state["playing"]:
            return []
        state["i"] = (state["i"] + state["speed"]) % len(starts)
        st = starts[state["i"]]
        window = emg[st:st + W].T.astype(np.float32)          # (C, W)
        level = np.clip(window.mean(1) / vmax, 0, 1)
        for b, h in zip(ring_bars, level[:8]):
            b.set_height(h)
        for b, h in zip(fe_bars, level[8:]):
            b.set_width(h)
        lo = max(0, st + W - hist)
        seg = emg[lo:st + W] / vmax
        for c, ln in enumerate(sig_lines):
            ln.set_data(np.arange(len(seg)), seg[:, c])

        s = scores_for(window)
        raw = int(np.argmax(s))
        state["votes"].append(raw)
        pred = Counter(state["votes"]).most_common(1)[0][0]
        true = int(lut[labels[st + W - 1]])
        state["n"] += 1; state["ok"] += int(pred == true)
        for k, b in enumerate(prob_bars):
            b.set_width(s[k])
            b.set_color("#2a6fdb" if k == pred else "#c9ced6")
        good = pred == true
        txt.set_text(f"Predicted: {names[pred]}")
        txt.set_color("#1e8f4e" if good else "#c0392b")
        sub.set_text(f"True: {names[true]}    |    running accuracy {state['ok'] / state['n']:.1%}"
                     f"    |    t = {st / C.FS:6.1f} s    |    smoothing {args.smooth} decisions")
        return []

    if args.save:
        anim = FuncAnimation(fig, update, frames=args.frames, interval=50, blit=False)
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        anim.save(args.save, writer="pillow", fps=20)
        print(f"Saved animation to {args.save}  (running accuracy {state['ok'] / max(state['n'], 1):.1%})")
        return

    ax_btn = fig.add_axes([0.03, 0.03, 0.08, 0.05])
    btn = Button(ax_btn, "Pause")
    ax_spd = fig.add_axes([0.15, 0.04, 0.17, 0.03])
    spd = Slider(ax_spd, "speed", 1, 10, valinit=1, valstep=1)

    def toggle(_):
        state["playing"] = not state["playing"]
        btn.label.set_text("Play" if not state["playing"] else "Pause")

    btn.on_clicked(toggle)
    spd.on_changed(lambda v: state.__setitem__("speed", int(v)))
    anim = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)  # noqa: F841
    plt.show()


if __name__ == "__main__":
    main()
