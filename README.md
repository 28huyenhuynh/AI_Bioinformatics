# EMG Gesture Classification for Prosthetic Limb Control

Bioinformatics Independent Study — Huynh Nhat Huyen & Le Nguyen Thanh Truc

Classical ML (LDA, kNN, linear SVM, Random Forest) vs deep learning (1D-CNN, LSTM, CNN-LSTM, 2D-CNN)
for sEMG gesture classification on **NinaPro DB1**, with a focus on **cross-subject generalisation**,
**robustness** to simulated real-world degradation, and a **muscle-channel (bioinformatics) analysis**.

---

## 1. Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python tests/test_pipeline.py        # 11 quick checks, no data needed
```

Data layout (already in this repo):

```
data/db1/s1/S1_A1_E1.mat  S1_A1_E2.mat  S1_A1_E3.mat
data/db1/s2/...
...
data/db1/s27/...
```

## 2. Project structure

```
semg/                  reusable library
  config.py            dataset constants: 100 Hz, channel map, global labels, gesture subsets
  data.py              load .mat -> global labels 0..52 -> clean sliding windows (+ cache)
  features.py          MAV, RMS, WL, SD + marginal Haar DWT (mDWT)
  classical.py         LDA, kNN3, kNN5, linear SVM, RF (scikit-learn pipelines)
  deep.py              CNN1D, LSTM, CNNLSTM, CNN2D (PyTorch) + training loop
  protocols.py         within-subject split, LOSO, metrics, leakage-safe fitting
  perturb.py           noise / fatigue / electrode-shift / channel-dropout simulation
  stats.py             Wilcoxon signed-rank + Holm correction, effect sizes
  bioinfo.py           activation profiles, variability, channel importance, elimination
  plots.py             all report figures
  sota.py              published deep models + their own paper recipes (Hartwell 2020, Hu 2018)
run_within_subject.py  STEP 1  reproduction (NinaPro standard repetition split)
run_sota.py            STEP 1b reproduction of published deep models (GPU; see notebooks/sota_colab.ipynb)
run_loso.py            STEP 2  leave-one-subject-out (main study)
run_stats.py           STEP 3  significance tests
run_bioinformatics.py  STEP 4  muscle-channel analysis
run_robustness.py      STEP 5  degradation curves
simulator.py           STEP 6  interactive replay GUI
tests/                 sanity tests for the pipeline rules
slurm/                 job scripts for the FUV AI GPU cluster
```

**Superseded first-pass scripts (September 2026).** Kept for the record; use the pipeline above instead.

| Script | Wrote to | Replaced by |
|---|---|---|
| `eda_db1.py` | `outputs/eda_db1_summary.txt`, `outputs/figures/` | `semg/data.py` + `run_bioinformatics.py` |
| `preprocess_db1.py` | `outputs/db1_preprocessed.npz` (deleted; rerun to rebuild) | `semg/data.py` (windows built from the `.mat` files directly) |
| `ml_baselines.py` | `outputs/ml_baselines_summary.txt`, confusion matrices in `outputs/figures/` | `run_within_subject.py`, `run_loso.py` |

`outputs/` is git-ignored and holds only those early figures and summaries. All current results go to `results/`.

## 3. Step by step

Every script prints progress, writes a `per_subject.csv` + figures under `results/`, and has `--help`.

### Step 1 — Reproduce published work (within-subject)
Train one model per subject on repetitions 1,3,4,6,8,9,10, test on 2,5,7 — the protocol of
Atzori et al. (2014). Compare our mean accuracy with the numbers in that paper's results table.

```bash
python run_within_subject.py --gestures all --models LDA,kNN5,SVM,RF          # ~20 min on a laptop
python run_within_subject.py --gestures all --models CNN2D,CNN1D --verbose     # deep reproduction
```
Output: `results/within_subject_all/` (summary.csv, per-subject bars, confusion matrices).
If our numbers are far from the paper, check window length/step and feature set before changing models.
`semg/deep.py::CNN2D` is a starting point for reproducing Atzori et al. (2016) — compare its layers with
that paper's architecture figure and adjust.

### Step 1b — Reproduce published deep models
`run_sota.py` trains each model with its own paper's windowing, split, training and metric, so the
numbers compare directly with the paper:

| `--paper` | Model | Protocol | Paper target (DB1) |
|---|---|---|---|
| `hartwell` | TtS CNN (754,933 params, Table I), Baseline CNN (Table III) | 150 ms windows, 10 ms step, 53 classes, 10 repetition splits, Adam 1e-3, 10 epochs | TtS macro-avg accuracy 66.6 ± 5.1 %, Baseline 65.0 ± 5.1 % |
| `hu` | Attention CNN-RNN, raw-image1 input | 200 ms windows, 52 movements, NinaPro split | 84.8 % per window, 96.5 % per trial (majority vote) |
| `jiang` | RIE: Inception + efficient channel attention (0.89 M params, matches paper) | 300 ms windows / 50 ms step, sym4 denoising, 52 movements, NinaPro split, Adam + cosine, 150 epochs | 88.27 % (current DB1 state of the art under the repetition split) |

```bash
python run_sota.py --paper hartwell --subjects 1 --splits 1     # one fold (~80 min on a laptop CPU)
python run_sota.py --paper hartwell                             # full protocol: use a GPU
python run_sota.py --paper hu
```
Full runs need a GPU. (connect NetBird; conda env `semg`):
```bash
mkdir -p logs
sbatch --array=1-27%1 slurm/sota.sbatch hartwell     # one subject per task
sbatch --array=1-27%1 slurm/sota.sbatch hu
squeue -u $USER; tail -f logs/sota_<jobid>_<task>.out
```
Without the cluster, open `notebooks/sota_colab.ipynb` in Google Colab. Finished folds are saved to
`results/sota_<paper>/folds/` and skipped on re-run, so a disconnected session simply resumes.
Settings a paper does not report (e.g. Hu et al. give no optimizer, learning rate or epochs) are listed
in `semg/sota.py::ASSUMPTIONS` and printed at the start of every run: state them in the report.

### Step 2 — Cross-subject study (LOSO)
```bash
python run_loso.py --models LDA,kNN3,kNN5,SVM,RF                 # -> results/loso_subset8 (~40 min, RF dominates)
python run_loso.py --models CNN1D,LSTM,CNNLSTM --out results/loso_deep --verbose
```
Default is the 8-gesture subset from the proposal: rest, open hand, fist, wrist flexion, wrist extension,
large-diameter (power) grasp, tip pinch, lateral grasp (`semg/config.py::SUBSET_8`).
Use `--gestures all` for all 52 movements. Use `--test-subjects 1,2,3` for a quick trial.

### Step 3 — Statistics
```bash
python run_stats.py --results results/loso_subset8 results/loso_deep --metric macro_f1
```
Pairwise Wilcoxon signed-rank tests on the 27 paired per-subject scores, Holm-corrected,
with rank-biserial effect sizes. This answers *"does the CNN significantly beat classical ML across subjects?"*

### Step 4 — Bioinformatics (muscle-channel analysis)
```bash
python run_bioinformatics.py --loso-results results/loso_subset8 --loso-model RF
python run_bioinformatics.py --folds 1,5,9,13,17,21,25     # faster: fewer LOSO folds for importance
```
| Output | Question it answers |
|---|---|
| `activation_heatmap.png` | Which channels/muscles does each gesture recruit? |
| `variability_heatmap.png` | Where do subjects differ most (CV across subjects)? |
| `channel_importance.csv/png` | Which channels matter (RF importance, F1 drop when removed)? |
| `backward_elimination.png` | Minimum channel subset for good control |
| `atypicality_vs_loso.png` | Do subjects with unusual muscle patterns generalise worse? |
| `correlations.txt` | Spearman correlations + anatomical sanity check (wrist flexion → FDS, extension → EDS) |

Caveat for the report: backward elimination chooses channels using the same LOSO folds it reports,
so the "minimum subset" score is slightly optimistic. To be strict, choose channels on some subjects
(`--folds`) and confirm the chosen subset with a separate `run_loso.py` run.

### Step 5 — Robustness
```bash
python run_robustness.py --models LDA,kNN5,SVM,RF
python run_robustness.py --models CNN1D,LSTM --out results/robustness_deep
python run_robustness.py --models RF,CNN1D --augment --out results/robustness_augmented   # mitigation
```
Each fold's model is trained once on clean data, then tested on the unseen subject under:
Gaussian noise (SNR 30→0 dB), fatigue-like amplitude gain (1.25→3×), electrode shift (forearm ring
rotated 1–4 positions), electrode loss (1–4 channels zeroed). Output: `degradation_curves.png`,
`robustness_table.csv`, `retention_harshest.csv`.

### Step 6 — Simulator
```bash
python simulator.py --subject 5 --model RF                  # interactive window (pause, speed)
python simulator.py --subject 5 --model CNN1D
python simulator.py --subject 5 --model RF --save results/simulator/demo.gif --frames 300
```
The model is trained without the replayed subject, so the demo behaves like fitting a prosthesis to a new user.
Running accuracy in the simulator is dominated by rest periods, so report LOSO numbers, not this one.

## 4. Methodological decisions

1. **No 20–450 Hz bandpass.** DB1 is sampled at 100 Hz (Nyquist 50 Hz) and its Otto Bock electrodes already output a rectified RMS envelope. For raw-EMG filtering use NinaPro DB2 (2 kHz).
2. **Global labels.** Each exercise file restarts labels at 1; offsets +0/+12/+29 give unique labels 1–52.
3. **Clean windows.** 200 ms windows (20 samples), 50 ms hop; a window is kept only if all its samples share one gesture and repetition. `restimulus`/`rerepetition` (corrected labels) are used.
4. **Rest balancing.** Rest windows are down-sampled to the median movement-class count per subject.
5. **No leakage.** Feature scaling / z-scoring is fitted on training data only; deep-learning early stopping uses held-out *training* subjects (LOSO) or repetition 10 (within-subject). The test subject is never seen before testing.
6. **Metrics.** Accuracy, balanced accuracy (= macro-averaged recall, the "macro accuracy" in the proposal), macro-F1, pooled confusion matrices.
7. **Linear SVM** uses exact LinearSVC on small training sets (within-subject) and SGD with hinge loss on large LOSO folds (>100k windows), where it matched LinearSVC macro-F1 while being ~25× faster.
8. **Fatigue** is simulated as an amplitude increase, not Gaussian noise; Gaussian noise is reported separately as sensor noise. **Electrode shift** is a rotation of the 8-electrode ring; channel dropout is reported separately as electrode failure.

## 5. Deep-learning notes
* GPU is used automatically if available (CUDA or Apple MPS); CPU works for the 8-gesture subset.
* Input: z-scored log-envelope windows, shape (batch, 10 channels, 20 samples).
* Adam (lr 1e-3, weight decay 1e-4), class-weighted cross-entropy, ReduceLROnPlateau, early stopping (patience 8).
* Try a 1-fold run first: `python run_loso.py --models CNN1D --test-subjects 1 --verbose`.

## 6. References
Dataset and baselines
* Atzori, M. et al. (2014). Electromyography data for non-invasive naturally-controlled robotic hand
  prostheses. *Scientific Data* 1, 140053. doi:10.1038/sdata.2014.53
* Atzori, M., Cognolato, M. & Müller, H. (2016). Deep learning with convolutional neural networks applied to
  electromyography data: a resource for the classification of movements for prosthetic hands.
  *Frontiers in Neurorobotics* 10, 9. doi:10.3389/fnbot.2016.00009

Reproduced deep models (`run_sota.py`)
* Hartwell, A., Kadirkamanathan, V. & Anderson, S. R. (2020). A temporal-to-spatial deep convolutional neural
  network for classification of hand movements from multichannel electromyography data. arXiv:2007.10879.
* Hu, Y., Wong, Y., Wei, W., Du, Y., Kankanhalli, M. & Geng, W. (2018). A novel attention-based hybrid CNN-RNN
  architecture for sEMG-based gesture recognition. *PLoS ONE* 13(10), e0206049. doi:10.1371/journal.pone.0206049
* Geng, W., Du, Y., Jin, W., Wei, W., Hu, Y. & Li, J. (2016). Gesture recognition by instantaneous surface EMG
  images. *Scientific Reports* 6, 36571. doi:10.1038/srep36571 — source of the training recipe used where
  Hu et al. report none.

Other DB1 state of the art discussed
* Wei, W., Dai, Q., Wong, Y., Hu, Y., Kankanhalli, M. & Geng, W. (2019). Surface-electromyography-based gesture
  recognition by multi-view deep learning. *IEEE Transactions on Biomedical Engineering* 66(10), 2964–2973.
  doi:10.1109/TBME.2019.2899222
* Xia, Y., Qiu, D., Zhang, C. & Liu, J. (2025). sEMG-based gesture recognition using multi-stream adaptive CNNs
  with integrated residual modules. *Frontiers in Bioengineering and Biotechnology* 13, 1487020.
  doi:10.3389/fbioe.2025.1487020 — 98.24 % on DB1 under a random window split (see leakage note in the report).

Robustness
* Pereira, J., Halatsis, D., Hodossy, B. & Farina, D. (2024). Tackling electrode shift in gesture recognition
  with HD-EMG electrode subsets. *ICASSP 2024*. doi:10.1109/ICASSP48485.2024.10448329 (arXiv:2401.02773) —
  needs a high-density grid, so on DB1 only its idea is used (ring-rotation augmentation, `--augment`).
