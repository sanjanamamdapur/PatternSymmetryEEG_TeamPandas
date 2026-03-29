"""
config.py
=========
Central configuration for all paths and analysis parameters.

BEFORE RUNNING ANY NOTEBOOK: set BIDS_ROOT to the folder on your machine
that contains ds004347. Everything else derives from that path.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
# Change this to your local path where ds004347 is stored.
BIDS_ROOT = Path(r"C:\Users\sanja\Documents\ds004347\ds004347")

# Results folder — created automatically if missing.
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Dataset info ───────────────────────────────────────────────────────────
TASK = "jacobsen"
N_SUBJECTS = 24
# Subject IDs that exist in the dataset (001 – 024).
# Subjects to skip (e.g. known recording failures) — fill in after inspection.
SUBJECTS_SKIP: list[str] = []

# ── Event mapping ──────────────────────────────────────────────────────────
# Trigger values as stored in the events.tsv 'value' column.
# Confirmed from sub-005_task-jacobsen_events.json:
#   x1 → Regular (reflectional symmetry)
#   x3 → Random
EVENT_ID = {"Regular": 1, "Random": 3}

# ── Channel configuration ──────────────────────────────────────────────────
# The dataset has 64 EEG channels + 8 external channels (EXG1–EXG8).
# EXG1–EXG4 are horizontal/vertical EOG (eye movements).
# EXG5–EXG8 are typically mastoid/reference channels in BIOSEMI setups.
# We rename them to conventional EOG names for ICA labelling.
EOG_CHANNEL_RENAME = {
    "EXG1": "EOG_hL",   # horizontal left
    "EXG2": "EOG_hR",   # horizontal right
    "EXG3": "EOG_vU",   # vertical upper
    "EXG4": "EOG_vD",   # vertical lower
    "EXG5": "M1",       # left mastoid (kept for ref comparison)
    "EXG6": "M2",       # right mastoid
    "EXG7": "EXG7",     # spare
    "EXG8": "EXG8",     # spare
}

# Channels to set as EOG type so MNE handles them correctly in ICA.
EOG_CHANNEL_NAMES = ["EOG_hL", "EOG_hR", "EOG_vU", "EOG_vD"]

# ── Filtering parameters ───────────────────────────────────────────────────
# High-pass at 0.1 Hz: removes slow DC drifts and linear trends without
# distorting ERP components. Lower values (0.01 Hz) are safer for slow
# components like the SPN (300–1000 ms) but risk DC drift contamination.
# Since P1/N1 are fast components (~80–200 ms), 0.1 Hz is safe here.
HIGHPASS_HZ = 0.1

# Low-pass at 40 Hz: well above the frequency content of P1/N1 (< 15 Hz)
# but removes high-frequency muscle noise. The original paper likely used
# a similar cutoff.
LOWPASS_HZ = 40.0

# Notch at 50 Hz: UK power line frequency (confirmed from eeg.json).
NOTCH_HZ = 50.0

# ── ICA parameters ─────────────────────────────────────────────────────────
# 64 channels → we can run up to 64 components, but 40 balances speed/coverage.
# ICLabel requires at least 15 components.
ICA_N_COMPONENTS = 40
ICA_RANDOM_STATE = 42   # for reproducibility

# ICLabel probability threshold for automatic rejection.
# Components with P(artefact) > this threshold are removed.
# 0.8 is conservative (keeps borderline components) — we will inspect visually.
ICLABEL_THRESHOLD = 0.8

# ── Epoching parameters ────────────────────────────────────────────────────
# −200 ms pre-stimulus baseline, 1000 ms post-stimulus.
# This captures the full P1 (~80–130 ms) and N1 (~150–200 ms) windows
# and extends to later components if needed.
EPOCH_TMIN = -0.2   # seconds
EPOCH_TMAX = 1.0    # seconds
BASELINE = (-0.2, 0.0)   # baseline correction window

# Peak-to-peak amplitude rejection threshold (µV).
# 100 µV is a standard conservative threshold for EEG epochs.
EPOCH_REJECT = {"eeg": 100e-6}   # MNE expects Volts, not µV

# Minimum number of accepted epochs per condition to keep a subject.
# Below this, the subject is flagged as an outlier.
MIN_EPOCHS_PER_CONDITION = 50  # out of 80 maximum

# ── Analysis parameters ────────────────────────────────────────────────────
# Electrodes of interest for P1/N1 — posterior occipital-parietal.
# PO7 and PO8 are the standard channels for early visual ERP components.
# These correspond to channels 25 & 62 in the authors' MATLAB scripts.
ROI_CHANNELS = ["PO7", "PO8"]

# P1 time window (ms) — positive deflection over occipital channels.
P1_WINDOW_MS = (80, 130)

# N1 time window (ms) — negative deflection following P1.
N1_WINDOW_MS = (150, 200)

# SPN window (ms) — sustained posterior negativity, used for optional extension.
SPN_WINDOW_MS = (300, 1000)

# ── Plotting defaults ──────────────────────────────────────────────────────
CONDITION_COLORS = {"Regular": "#E24B4A", "Random": "#378ADD"}
FIGURE_DPI = 150
