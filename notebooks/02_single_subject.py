# %% [markdown]
# # Notebook 02 — Single Subject Pipeline (Milestone 3)
#
# **Goal:** Run the complete preprocessing and analysis pipeline on sub-005
# and verify each step visually before applying it to all 24 subjects.
#
# This notebook documents every parameter decision with a motivation sentence,
# as required by the grading criteria.

# %%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from src.config import (
    RESULTS_DIR, EVENT_ID, ROI_CHANNELS,
    ICA_N_COMPONENTS, ICLABEL_THRESHOLD,
    EPOCH_TMIN, EPOCH_TMAX, BASELINE, EPOCH_REJECT,
    P1_WINDOW_MS, N1_WINDOW_MS,
)

SUBJECT = "005"
print(f"Running full pipeline for sub-{SUBJECT}")

# %% [markdown]
# ## Step 1: Load raw data

# %%
from src.preprocessing import load_raw

raw = load_raw(SUBJECT)
print(raw.info)

# Sanity check: confirm channel types after fixes
n_eeg  = len(mne.pick_types(raw.info, eeg=True))
n_eog  = len(mne.pick_types(raw.info, eog=True))
n_misc = len(mne.pick_types(raw.info, misc=True))
print(f"\nChannel types: {n_eeg} EEG, {n_eog} EOG, {n_misc} misc")
assert n_eeg == 64,  f"Expected 64 EEG channels, got {n_eeg}"
assert n_eog == 4,   f"Expected 4 EOG channels, got {n_eog}"
assert n_misc == 4,  f"Expected 4 misc channels (M1, M2, EXG7, EXG8), got {n_misc}"
print("Channel type sanity check passed.")

# %% [markdown]
# ## Step 2: Filter
#
# **Motivation:** 0.1 Hz high-pass removes slow DC drifts without distorting
# slow components like the SPN (300-1000 ms). A 1 Hz cutoff would distort the
# SPN; 0.1 Hz is the safer choice for this dataset. The 40 Hz low-pass removes
# muscle noise -- P1/N1 are well below 15 Hz so this does not affect our signal.
# The 50 Hz notch removes UK power line noise (confirmed from eeg.json).

# %%
from src.preprocessing import filter_raw
from src.plotting import plot_power_spectrum

print("PSD before filtering:")
fig_pre = plot_power_spectrum(raw, f"{SUBJECT}_pre-filter")
plt.show()

raw_filt = filter_raw(raw)

print("PSD after filtering:")
fig_post = plot_power_spectrum(raw_filt, f"{SUBJECT}_post-filter")
plt.show()

# %% [markdown]
# **Observation:** [Fill in after looking at the plots.]
# - Is the 50 Hz peak visible before filtering and reduced after?
# - Does the 1/f slope (power decreasing with frequency) look normal?

# %% [markdown]
# ## Step 3: ICA — fit
#
# **Motivation:** ICA decomposes EEG into statistically independent components.
# Eye blinks produce a characteristic frontal topography. ICA finds these and
# allows us to subtract them without distorting signals at other channels.
#
# We fit on an internal 1-100 Hz, average-referenced, EEG-only copy (raw_ica)
# as required by ICLabel. The ICA solution is applied to the 0.1-40 Hz data
# so slow components are preserved. Algorithm: extended Infomax, which is
# required by ICLabel (FastICA degrades classification accuracy).

# %%
from src.preprocessing import run_ica

print(f"Fitting ICA ({ICA_N_COMPONENTS} components, extended Infomax)...")
print("This takes 1-3 minutes...")
# IMPORTANT: pass raw (unfiltered), NOT raw_filt.
# run_ica builds raw_ica by filtering 1-100 Hz from scratch.
# If we pass raw_filt (already lowpassed at 40 Hz), the content
# above 40 Hz is gone and ICLabel gets out-of-spec data.
ica, raw_ica = run_ica(raw)          # run_ica returns (ica, raw_ica)
print("ICA fitting complete.")

# Quick check: raw_ica should be EEG-only, 64 channels
n_ica_ch = len(mne.pick_types(raw_ica.info, eeg=True))
print(f"raw_ica has {n_ica_ch} EEG channels (expected 64 for ICLabel).")
assert n_ica_ch == 64, f"ICLabel needs 64 channels, got {n_ica_ch}"

# %% [markdown]
# ## Step 4: ICA — label with ICLabel

# %%
from src.preprocessing import label_ica_components, select_artefact_components

# label_ica_components receives raw_ica (the 1-100 Hz, avg-ref, EEG-only copy)
ic_labels = label_ica_components(raw_ica, ica)

# Display the classification table (only if ICLabel succeeded)
if ic_labels is not None:
    proba = ic_labels["y_pred_proba"]
    # Defensive shape check: ICLabel should return (n_components, 7).
    # A 1D array means ICLabel received out-of-spec data (usually the
    # 40 Hz lowpass issue). With the run_ica(raw) fix this should not
    # happen, but we handle it gracefully just in case.
    if proba.ndim == 2 and proba.shape[1] == 7:
        labels_df = pd.DataFrame({
            "label":           ic_labels["labels"],
            "prob_brain":      proba[:, 0],
            "prob_muscle":     proba[:, 1],
            "prob_eye":        proba[:, 2],
            "prob_heart":      proba[:, 3],
            "prob_line_noise": proba[:, 4],
            "prob_chan_noise":  proba[:, 5],
            "prob_other":      proba[:, 6],
        }).round(3)
        print(labels_df.head(20).to_string())
    else:
        print(f"WARNING: y_pred_proba has unexpected shape {proba.shape}.")
        print("ICLabel may have received out-of-spec data.")
        print("Labels:", ic_labels["labels"])
else:
    print("ICLabel unavailable -- will use EOG-correlation fallback.")

# %%
# select_artefact_components receives raw_filt (not raw_ica) because
# find_bads_eog needs the EOG channels which were dropped in raw_ica.
exclude_indices = select_artefact_components(ica, ic_labels, raw_filt)
print(f"\nComponents selected for removal: {exclude_indices}")

if ic_labels is not None and exclude_indices:
    print(f"Labels: {[ic_labels['labels'][i] for i in exclude_indices]}")
elif not exclude_indices:
    print("No components selected -- ICA will not change the data.")
    print("Note: this is unusual. Check ICLabel warnings above.")

# %% [markdown]
# ## Step 5: ICA — visual inspection
#
# **Critical check before removing components:**
# 1. Do removed components have a frontal topography (eye) or
#    temporal/edge topography (muscle)?
# 2. Do their time courses show blink spikes or sustained muscle bursts?
# 3. Are we accidentally removing posterior occipital components
#    (which would be genuine visual brain activity)?
#
# If any removed component looks like brain signal, increase ICLABEL_THRESHOLD
# in config.py and re-run.

# %%
from src.plotting import plot_ica_components, plot_ica_overlay

fig_comp = plot_ica_components(ica, raw_filt, exclude_indices, SUBJECT, n_components=20)
plt.show()

# %% [markdown]
# **Observation for each removed component:**
# - Component X: [frontal topography, large blink spikes] -> eye artefact, correct to remove.
# - Component Y: [temporal topography, high-freq bursts] -> muscle artefact, correct to remove.
#
# "This seems correct because..."

# %%
fig_overlay = plot_ica_overlay(raw_filt, ica, SUBJECT)
plt.show()

# %% [markdown]
# **Observation:** The overlay shows the removed signal (artefact waveform).
# It should look like blinks / muscle bursts, not like smooth ERP waveforms.

# %% [markdown]
# ## Step 6: Apply ICA, interpolate bad channels, re-reference

# %%
from src.preprocessing import (
    apply_ica, find_bad_channels, interpolate_bad_channels, set_average_reference
)

raw_clean = apply_ica(raw_filt, ica, exclude_indices)

bad_channels = find_bad_channels(raw_clean)
print(f"Bad channels detected: {bad_channels or 'none'}")
raw_interp = interpolate_bad_channels(raw_clean, bad_channels)

raw_preprocessed = set_average_reference(raw_interp)
print("Average reference applied.")

# %% [markdown]
# **Motivation for average reference:**
# We chose average reference over the authors' likely linked-mastoid reference
# because our 64-channel cap provides sufficient spatial sampling for the
# average to approximate electrical zero (Bertrand et al., 1985). This is a
# deliberate pipeline difference to test robustness of the P1/N1 findings.

# %% [markdown]
# ## Step 7: Epoch
#
# **Motivation:** Epochs cut from -200 to +1000 ms capture the full P1/N1
# (80-200 ms) and the later SPN (300-1000 ms). The -200 ms pre-stimulus
# baseline is long enough for a stable mean estimate for baseline correction.

# %%
from src.epoching import create_epochs, drop_bad_epochs

epochs = create_epochs(raw_preprocessed)
print(f"Epochs created: {len(epochs)} total "
      f"({len(epochs['Regular'])} Regular, {len(epochs['Random'])} Random)")

# Sanity check: event counts should match the raw file
assert len(epochs['Regular']) <= 80, "More Regular epochs than expected."
assert len(epochs['Random'])  <= 80, "More Random epochs than expected."

# %%
epochs_clean, rejection_log = drop_bad_epochs(epochs)

print("\nRejection summary:")
for cond, log in rejection_log.items():
    print(f"  {cond}: {log['kept']}/{log['original']} kept ({log['percent_kept']}%)")

# %% [markdown]
# **Observation:** [How many epochs were rejected?]
# "A rejection rate of X% seems [reasonable/high] because..."
# If >30% rejected in any condition, re-inspect the raw data for
# residual artefacts before proceeding.

# %% [markdown]
# ## Step 8: Epoch image sanity check
#
# Each row = one trial, colour = amplitude. Residual artefacts appear as
# bright or dark rows. Any systematic pattern visible across trials would
# suggest a non-neural signal is contaminating the data.

# %%
fig = epochs_clean["Regular"].plot_image(
    picks=ROI_CHANNELS[0], show=False,
    title=f"Regular epochs at {ROI_CHANNELS[0]} -- sub-{SUBJECT}",
)
plt.show()

fig = epochs_clean["Random"].plot_image(
    picks=ROI_CHANNELS[0], show=False,
    title=f"Random epochs at {ROI_CHANNELS[0]} -- sub-{SUBJECT}",
)
plt.show()

# %% [markdown]
# ## Step 9: Compute evoked responses

# %%
from src.epoching import compute_evokeds, compute_difference_wave

evokeds   = compute_evokeds(epochs_clean)
diff_wave = compute_difference_wave(evokeds)

for cond, ev in evokeds.items():
    print(f"{cond}: {ev.nave} trials averaged")

# %% [markdown]
# ## Step 10: ERP waveforms
#
# Key result for this subject. P1 (~100 ms) should be a positive peak at
# PO7/PO8. N1 (~170 ms) should be a negative peak. If the paper's findings
# are robust, the Regular condition should show larger P1 and/or N1 than Random.

# %%
from src.plotting import plot_erp_waveforms, plot_difference_wave as plot_diff

fig_erp  = plot_erp_waveforms(evokeds, subject=SUBJECT)
plt.show()

fig_diff = plot_diff(diff_wave, subject=SUBJECT)
plt.show()

# %% [markdown]
# **Observation:**
# - "P1 peak is visible at approximately [X] ms with amplitude [Y] uV."
# - "The Regular condition shows [larger/similar/smaller] P1 vs Random."
# - "This [is/is not] consistent with the paper's finding because..."

# %% [markdown]
# ## Step 11: Extract metrics

# %%
from src.analysis import extract_subject_metrics

metrics = extract_subject_metrics(evokeds, SUBJECT)
print("\nAmplitude metrics (mean over PO7/PO8):")
for key, val in metrics.items():
    if key != "subject":
        print(f"  {key}: {val:.3f} uV")

# %% [markdown]
# ## Step 12: Topographic maps

# %%
from src.plotting import plot_topomap_series, plot_difference_topomap

fig_topo_reg  = plot_topomap_series(evokeds, condition="Regular")
plt.show()

fig_topo_ran  = plot_topomap_series(evokeds, condition="Random")
plt.show()

fig_topo_diff = plot_difference_topomap(evokeds)
plt.show()

# %% [markdown]
# **Observation:**
# "The P1 window topomap shows [posterior/frontal/diffuse] distribution.
# A posterior distribution would be consistent with primary visual cortex
# processing, supporting the paper's claims."

# %% [markdown]
# ## Pipeline summary
#
# | Step | Input | Output | Key decision |
# |---|---|---|---|
# | Load | BDF + BIDS sidecars | Raw (64 EEG, 4 EOG, 4 misc) | Strip quotes, type M1/M2 as misc |
# | Filter | Raw | Filtered raw | 0.1-40 Hz + 50 Hz notch |
# | ICA | Filtered | ICA object + raw_ica | Extended Infomax, 40 components |
# | ICLabel | raw_ica | Component labels | Threshold 0.8, + EOG correlation |
# | Apply ICA | Filtered | Artefact-free raw | Subtract eye/muscle components |
# | Bad channels | Artefact-free | Interpolated | SD threshold (0.01x, 5x median) |
# | Re-reference | Interpolated | Avg-ref | Average (not mastoid) |
# | Epoch | Avg-ref | Epochs | -200 to +1000 ms, baseline -200:0 |
# | Reject | Epochs | Clean epochs | +-100 uV peak-to-peak |
# | Average | Clean epochs | Evokeds | Per condition |

# %%
print(f"\nNotebook 02 complete for sub-{SUBJECT}. Figures saved to: {RESULTS_DIR}")