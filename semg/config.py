"""Project-wide constants for NinaPro DB1.

Everything that describes the dataset (sampling rate, channel layout, label
numbering, gesture subsets) lives here so every script uses the same values.
"""
import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "db1")
CACHE_DIR = os.path.join(ROOT, "cache")
RESULTS_DIR = os.path.join(ROOT, "results")

# ---------------------------------------------------------------------------
# Signal
# ---------------------------------------------------------------------------
# DB1 uses Otto Bock 13E200 electrodes, which output an already rectified and
# RMS-smoothed envelope sampled at 100 Hz. Nyquist is 50 Hz, so a 20-450 Hz
# bandpass is NOT possible on this dataset; we do no bandpass filtering.
FS = 100
N_CHANNELS = 10

# Channels 1-8 (index 0-7) are equally spaced in a ring around the forearm at
# the height of the radio-humeral joint. Channel 9 sits on flexor digitorum
# superficialis, channel 10 on extensor digitorum superficialis (DB1 docs).
RING_CHANNELS = list(range(8))
CHANNEL_NAMES = [f"ch{i + 1}" for i in range(N_CHANNELS)]
CHANNEL_ANNOTATION = {
    **{f"ch{i + 1}": f"Forearm ring, position {i + 1}/8 (radio-humeral level)" for i in range(8)},
    "ch9": "Flexor digitorum superficialis (FDS)",
    "ch10": "Extensor digitorum superficialis (EDS)",
}

# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
# Each exercise file numbers its movements from 1. Add these offsets to get the
# global 1..52 numbering used in Atzori et al. (2014). 0 is always rest.
EXERCISE_OFFSETS = {1: 0, 2: 12, 3: 29}
EXERCISE_SIZES = {1: 12, 2: 17, 3: 23}

# Movement names following Atzori et al. (2014), Scientific Data 1:140053.
# Double-check against the movement figure on the NinaPro website before you
# quote them in the report.
GESTURE_NAMES = {
    0: "Rest",
    # Exercise 1: basic finger movements
    1: "Index flexion", 2: "Index extension", 3: "Middle flexion",
    4: "Middle extension", 5: "Ring flexion", 6: "Ring extension",
    7: "Little flexion", 8: "Little extension", 9: "Thumb adduction",
    10: "Thumb abduction", 11: "Thumb flexion", 12: "Thumb extension",
    # Exercise 2: hand postures and wrist movements
    13: "Thumb up", 14: "Index+middle extension", 15: "Ring+little flexion",
    16: "Thumb opposing little", 17: "Abduction all fingers (open hand)",
    18: "Fingers flexed (fist)", 19: "Pointing index", 20: "Adduction extended fingers",
    21: "Supination (middle axis)", 22: "Pronation (middle axis)",
    23: "Supination (little axis)", 24: "Pronation (little axis)",
    25: "Wrist flexion", 26: "Wrist extension", 27: "Radial deviation",
    28: "Ulnar deviation", 29: "Wrist extension, closed hand",
    # Exercise 3: grasps and functional movements
    30: "Large diameter grasp", 31: "Small diameter grasp", 32: "Fixed hook grasp",
    33: "Index finger extension grasp", 34: "Medium wrap", 35: "Ring grasp",
    36: "Prismatic four fingers", 37: "Stick grasp", 38: "Writing tripod",
    39: "Power sphere", 40: "Three finger sphere", 41: "Precision sphere",
    42: "Tripod grasp", 43: "Prismatic pinch", 44: "Tip pinch",
    45: "Quadpod grasp", 46: "Lateral grasp", 47: "Parallel extension grasp",
    48: "Extension type grasp", 49: "Power disk", 50: "Open bottle (tripod)",
    51: "Turn screw", 52: "Cut with knife",
}

# Reduced 8-class set from the proposal: rest, open hand, fist, power grip,
# tip pinch, lateral grasp, wrist flexion, wrist extension.
SUBSET_8 = [0, 17, 18, 25, 26, 30, 44, 46]
ALL_GESTURES = list(range(53))  # 52 movements + rest

GESTURE_SETS = {"subset8": SUBSET_8, "all": ALL_GESTURES}

# ---------------------------------------------------------------------------
# Default experiment settings
# ---------------------------------------------------------------------------
WINDOW = 20          # samples -> 200 ms at 100 Hz
STEP = 5             # samples -> 50 ms hop
REST_RATIO = 1.0     # keep rest windows = ratio * median count of other classes
SEED = 42

# NinaPro recommended within-subject split for DB1 (10 repetitions).
TRAIN_REPS = [1, 3, 4, 6, 8, 9, 10]
TEST_REPS = [2, 5, 7]

CLASSICAL_MODELS = ["LDA", "kNN3", "kNN5", "SVM", "RF", "SVM-exact"]
DEEP_MODELS = ["CNN1D", "LSTM", "CNNLSTM", "CNN2D"]


def gesture_name(label):
    return GESTURE_NAMES.get(int(label), str(label))
