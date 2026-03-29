# %% [markdown]
# # Notebook 01 — Data Inspection (Milestone 2)
#
# **Goal:** Load a single subject, inspect the raw signal, verify event
# mapping, and document the first impression of the data.
#
# **Subject used:** sub-005 (chosen arbitrarily as a representative subject).
#
# This notebook satisfies Milestone 2 requirements:
# - Show a first impression of the continuous data of one subject.
# - Map out the required analysis steps.
# - Identify dataset-specific quirks.

# %%
import sys
import os

# Make sure the project root is on the path so we can import from src/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

# Print software versions for reproducibility.
mne.sys_info()

# %%
from src.config import BIDS_ROOT, EVENT_ID, ROI_CHANNELS, RESULTS_DIR
#from src.preprocessing import load_raw
from src.preprocessing import load_raw, make_bids_path, _load_events_from_tsv
# Check that BIDS_ROOT is set correctly.
assert BIDS_ROOT.exists(), (
    f"BIDS_ROOT '{BIDS_ROOT}' does not exist. "
    "Please edit src/config.py and set BIDS_ROOT to your local ds004347 path."
)
print(f"BIDS root: {BIDS_ROOT}")

# %% [markdown]
# ## 1. Load subject 005

# %%
SUBJECT = "005"
raw = load_raw(SUBJECT, preload=True)

print(f"\nSubject: {SUBJECT}")
print(f"Sampling frequency: {raw.info['sfreq']} Hz")
print(f"Recording duration: {raw.times[-1]:.1f} s")
print(f"Total channels: {len(raw.ch_names)}")
print(f"EEG channels: {len(mne.pick_types(raw.info, eeg=True))}")
print(f"EOG channels: {len(mne.pick_types(raw.info, eog=True))}")

# %% [markdown]
# ## 2. First impression: raw signal
#
# We look for:
# - Flat channels (dead electrodes).
# - Channels with very large amplitude (noisy/bridged).
# - 50 Hz line noise (periodic oscillation before filtering).
# - Eye blinks visible in EOG channels.

# %%
from src.plotting import plot_raw_overview

fig = plot_raw_overview(raw, SUBJECT, duration=20.0)
plt.show()

# %% [markdown]
# **Observation (fill in after looking at the plot):**
# "This seems correct because... / This is strange because..."
#
# Typical things to note:
# - Which channels look noisy or flat?
# - Is line noise visible?
# - Are eye blinks clearly visible in EOG channels?

# %% [markdown]
# ## 3. Power spectral density
#
# Checking the PSD confirms the 1/f structure of EEG and helps identify
# whether line noise at 50 Hz is present before filtering.

# %%
from src.plotting import plot_power_spectrum

fig = plot_power_spectrum(raw, SUBJECT)
plt.show()

# %% [markdown]
# **Observation:** If the 50 Hz peak is visible here (before filtering),
# the notch filter in our preprocessing pipeline will remove it.

# %% [markdown]
# ## 4. Event inspection and condition mapping
#
# The events.tsv 'value' column contains triggers:
# - value = 1 → Regular (reflectional symmetry pattern)
# - value = 3 → Random (no symmetry)
#
# The 'trial_type' column is entirely n/a, so we MUST use 'value' for mapping.
#
# Noted quirk: events.json states onset units as "ms", but the actual
# values (e.g. 12.87) are clearly in seconds. MNE-BIDS follows BIDS spec
# and treats them as seconds, which is correct.

# %%
bids_path = make_bids_path(SUBJECT)
events = _load_events_from_tsv(bids_path, raw)

print("Events shape:", events.shape)
print(f"\nTotal events found: {len(events)}")
for cond, val in EVENT_ID.items():
    n = np.sum(events[:, 2] == val)
    print(f"  {cond} (value={val}): {n} trials")

# Sanity check: expect 80 trials per condition.
assert np.sum(events[:, 2] == 1) == 80, "Expected 80 Regular trials."
assert np.sum(events[:, 2] == 3) == 80, "Expected 80 Random trials."
print("\n✓ Sanity check passed: 80 trials per condition as expected.")

# %% [markdown]
# ## 5. Check inter-trial intervals
#
# We inspect the time between events to confirm the stimulus presentation
# rate matches what the experiment script (EEG14.py) implies.
# The experiment used a 3 s stimulus + 1 s ITI ≈ 4–5 s per trial.

# %%
event_onsets_s = events[:, 0] / raw.info["sfreq"]
itis = np.diff(event_onsets_s)

print(f"Inter-trial interval (stimulus onset to onset):")
print(f"  Mean:  {itis.mean():.2f} s")
print(f"  Std:   {itis.std():.2f} s")
print(f"  Min:   {itis.min():.2f} s")
print(f"  Max:   {itis.max():.2f} s")

fig, ax = plt.subplots(figsize=(8, 3))
ax.hist(itis, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
ax.set_xlabel("Inter-trial interval (s)")
ax.set_ylabel("Count")
ax.set_title(f"ITI distribution — sub-{SUBJECT}")
plt.tight_layout()
fig.savefig(RESULTS_DIR / f"sub-{SUBJECT}_iti_distribution.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# **Observation:** Most ITIs should be approximately 4–6 s based on the
# PsychoPy script (3 s stimulus + 1 s ITI + question screen).
# Longer ITIs (e.g. >8 s) correspond to rest blocks inserted every
# 30 trials in the original script.

# %% [markdown]
# ## 6. Channel layout
#
# Confirm the 64-channel biosemi64 layout is applied correctly.
# PO7 and PO8 (our channels of interest) should be at posterior-occipital
# locations on the left and right hemispheres.

# %%
fig = raw.plot_sensors(show_names=True, show=False)
fig.set_size_inches(8, 8)

# Highlight ROI channels.
# (Manual: look for PO7 and PO8 in the plot — they should be at the back)
plt.title(f"Electrode layout — sub-{SUBJECT}\nROI channels: {ROI_CHANNELS}")
fig.savefig(RESULTS_DIR / f"sub-{SUBJECT}_electrode_layout.png", bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Summary of data inspection
#
# Fill in this section after looking at the plots above.
#
# | Check | Result |
# |---|---|
# | Onset unit bug (ms vs s) | values are in seconds, json says ms — loading is correct |
# | Event counts | 80 Regular, 80 Random ✓ |
# | Obvious bad channels | [fill in] |
# | 50 Hz line noise visible | [fill in] |
# | Eye blinks visible in EOG | [fill in] |
# | ITI consistent with experiment design | [fill in] |
# | PO7/PO8 visible in posterior location | [fill in] |
#
# **Overall first impression:**
# "The data looks [clean/noisy] because..."

# %%
print("Notebook 01 complete. Figures saved to:", RESULTS_DIR)
