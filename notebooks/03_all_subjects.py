# %% [markdown]
# # Notebook 03 — All Subjects (Milestone 4)
#
# **Goal:** Run the preprocessing and analysis pipeline across all 24 subjects,
# compute grand averages, identify outliers, and run group-level statistics.
#
# **Milestones covered:**
# - Milestone 4: Pipeline working for all subjects, outliers identified.
# - Final report: Grand-average ERPs, topomaps, difference wave, statistics.

# %%
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from src.config import (
    RESULTS_DIR, N_SUBJECTS, SUBJECTS_SKIP, EVENT_ID,
    ROI_CHANNELS, MIN_EPOCHS_PER_CONDITION,
)

# Build list of all subject IDs (001 to 024, zero-padded to 3 digits).
ALL_SUBJECTS = [f"{i:03d}" for i in range(1, N_SUBJECTS + 1)
                if f"{i:03d}" not in SUBJECTS_SKIP]
print(f"Subjects to process: {ALL_SUBJECTS}")
print(f"Skipped: {SUBJECTS_SKIP}")

# %% [markdown]
# ## 1. Run full pipeline for all subjects
#
# We store:
# - `all_evokeds`: per-condition evoked responses per subject (for grand average).
# - `all_metrics`: amplitude metrics per subject (for statistics).
# - `all_rejection_logs`: epoch rejection rates per subject (for QC summary).
# - `outlier_subjects`: subjects who failed quality checks.

# %%
from src.preprocessing import preprocess_subject
from src.epoching import run_epoching_pipeline
from src.analysis import extract_subject_metrics

# Storage.
all_evokeds      = {"Regular": [], "Random": []}
all_diff_waves   = []
all_metrics      = []
all_rejection_logs = {}
outlier_subjects = []
failed_subjects  = []

for subj in ALL_SUBJECTS:
    try:
        # --- Preprocessing ---
        raw_preprocessed, ica, exclude_idx, bad_chs = preprocess_subject(
            subj, save_ica=True
        )

        # --- Epoching ---
        epochs_clean, evokeds, diff_wave, rejection_log, is_ok = run_epoching_pipeline(
            raw_preprocessed, subj
        )

        # --- Store results ---
        all_rejection_logs[subj] = rejection_log

        if not is_ok:
            outlier_subjects.append(subj)
            print(f"  ⚠  sub-{subj} added to outliers (low epoch count).")
            # We still keep them in the analysis for now but flag them.
            # The final decision to exclude them will be discussed in the report.

        for cond in EVENT_ID:
            all_evokeds[cond].append(evokeds[cond])
        all_diff_waves.append(diff_wave)

        metrics = extract_subject_metrics(evokeds, subj)
        all_metrics.append(metrics)

        print(f"  ✓  sub-{subj} done.\n")

    except Exception as e:
        print(f"  ✗  sub-{subj} FAILED: {e}\n")
        failed_subjects.append(subj)

print(f"\n{'='*50}")
print(f"Completed: {len(all_metrics)} subjects")
print(f"Outliers:  {outlier_subjects}")
print(f"Failed:    {failed_subjects}")

# %% [markdown]
# **Observation:** [Which subjects failed or were flagged?
# What is the typical rejection rate? Does it vary systematically
# across the session (e.g. more rejections at the end due to fatigue)?]

# %% [markdown]
# ## 2. Epoch rejection summary

# %%
from src.analysis import build_rejection_summary
from src.plotting import plot_epoch_rejection_summary

rejection_df = build_rejection_summary(all_rejection_logs)
print(rejection_df.to_string())

fig_rejection = plot_epoch_rejection_summary(rejection_df, outlier_subjects)
plt.show()

# Save summary to CSV.
rejection_df.to_csv(RESULTS_DIR / "epoch_rejection_summary.csv")
print(f"Rejection summary saved.")

# %% [markdown]
# **Observation:** [Fill in after looking at the bar chart.]
# "Subjects [X, Y, Z] have notably low retention rates. This could be
# because... We will [include/exclude] them based on..."

# %% [markdown]
# ## 3. Build metrics DataFrame and export

# %%
from src.analysis import build_metrics_dataframe

metrics_df = build_metrics_dataframe(all_metrics)
print(metrics_df.round(3).to_string())

metrics_df.to_csv(RESULTS_DIR / "amplitude_metrics.csv")
print("\nMetrics saved to amplitude_metrics.csv")

# %% [markdown]
# ## 4. Grand averages

# %%
from src.analysis import compute_grand_averages

grand_averages = compute_grand_averages(all_evokeds)

# Also compute grand-average difference wave.
grand_diff = mne.grand_average(all_diff_waves)
grand_diff.comment = "Regular - Random (grand average)"

# %% [markdown]
# ## 5. Grand-average ERP waveforms
#
# This is the key group-level result, equivalent to what the original paper
# shows. We plot conditions overlaid at PO7 and PO8 (the ROI).

# %%
from src.plotting import plot_erp_waveforms, plot_difference_wave as plot_diff

fig_grand_erp = plot_erp_waveforms(grand_averages, subject="grand_average")
plt.show()

# %% [markdown]
# **Observation:**
# "The grand-average ERP shows a P1 component peaking at approximately [X] ms.
# The Regular condition shows [larger/smaller] P1 amplitude compared to Random.
# This [is/is not] consistent with the paper's finding of P1 modulation by symmetry."

# %%
fig_grand_diff = plot_diff(grand_diff, subject="grand_average")
plt.show()

# %% [markdown]
# **Observation:**
# "The grand-average difference wave shows [description].
# The [presence/absence] of a posterior negative deflection in the SPN
# window (300–1000 ms) [supports/does not support] the paper's SPN finding."

# %% [markdown]
# ## 6. Topographic maps of grand average

# %%
from src.plotting import plot_topomap_series, plot_difference_topomap

fig_topo_reg = plot_topomap_series(grand_averages, condition="Regular")
plt.show()

fig_topo_ran = plot_topomap_series(grand_averages, condition="Random")
plt.show()

fig_topo_diff = plot_difference_topomap(grand_averages)
plt.show()

# %% [markdown]
# **Observation:**
# "The topomap in the P1 window (80–130 ms) shows a [posterior/frontal/diffuse]
# distribution, which [is/is not] consistent with early visual cortex processing.
# The SPN difference topomap in 300–1000 ms shows [description], which
# [matches/differs from] the expected posterior negativity."

# %% [markdown]
# ## 7. Per-subject amplitude distributions

# %%
from src.plotting import plot_amplitude_distributions

for component in ["P1", "N1", "SPN"]:
    fig = plot_amplitude_distributions(metrics_df, component=component)
    plt.show()

# %% [markdown]
# ## 8. Group statistics

# %%
from src.analysis import run_all_stats
from src.plotting import plot_stats_table

stats_table = run_all_stats(metrics_df)
print("\nGroup statistics (paired t-tests, Regular vs Random at PO7/PO8):")
print(stats_table.round(3).to_string())

stats_table.to_csv(RESULTS_DIR / "group_statistics.csv")

fig_stats = plot_stats_table(stats_table)
plt.show()

# %% [markdown]
# **Interpretation:**
# - P1: "The [significant/non-significant] difference (t=[X], p=[Y], d=[Z])
#   [supports/does not support] the paper's finding that P1 is modulated by
#   symmetry. A Cohen's d of [Z] indicates a [small/medium/large] effect."
# - N1: [similar interpretation]
# - SPN: [similar interpretation]
#
# "Taken together, these results [suggest/do not suggest] that our alternative
# pipeline reproduces the main findings of the paper. The [convergence/divergence]
# between pipelines indicates that the effect is [robust/fragile] to analysis
# choices."

# %% [markdown]
# ## 9. Outlier analysis
#
# We investigate the outlier subjects identified earlier.

# %%
if outlier_subjects:
    print(f"Outlier subjects: {outlier_subjects}")
    print("\nTheir rejection rates:")
    print(rejection_df.loc[[s for s in outlier_subjects if s in rejection_df.index]])

    # Re-run stats excluding outliers and compare.
    good_subjects = [s for s in metrics_df.index if s not in outlier_subjects]
    metrics_df_clean = metrics_df.loc[good_subjects]

    stats_clean = run_all_stats(metrics_df_clean)
    print(f"\nStats WITHOUT outliers ({len(metrics_df_clean)} subjects):")
    print(stats_clean.round(3).to_string())

    print(f"\nStats WITH outliers ({len(metrics_df)} subjects):")
    print(stats_table.round(3).to_string())

    # If p-values change substantially, the outliers have a strong influence
    # and we should report both analyses in the paper.
else:
    print("No outlier subjects detected.")

# %% [markdown]
# **Observation:**
# "Excluding outlier subjects [changed/did not substantially change] the
# statistical results. The p-value for P1 changed from [X] to [Y].
# We therefore [include/exclude] these subjects in the final analysis and
# report both results transparently."

# %% [markdown]
# ## 10. Final pipeline summary for report
#
# This table gives an at-a-glance overview of our pipeline vs the authors'.

# %%
pipeline_comparison = pd.DataFrame({
    "Step": [
        "Software", "EEG reference", "High-pass filter", "Low-pass filter",
        "Notch filter", "Artefact removal", "Epoch window", "Baseline",
        "Epoch rejection", "Statistics",
    ],
    "Authors (MATLAB/EEGLAB)": [
        "EEGLAB", "Linked mastoids (likely)", "Unknown", "Unknown",
        "Assumed 50 Hz", "Manual ICA inspection", "Unknown", "200 ms",
        "Manual", "t-test on mean amplitude",
    ],
    "Ours (MNE-Python)": [
        "MNE-Python 1.7.1", "Average reference", "0.1 Hz (FIR Hamming)",
        "40 Hz (FIR Hamming)", "50 Hz notch filter", "Automated ICLabel (p>0.8)",
        "−200 to +1000 ms", "−200 to 0 ms", "±100 µV peak-to-peak",
        "Paired t-test + Cohen's d",
    ],
})
print(pipeline_comparison.to_string(index=False))
pipeline_comparison.to_csv(RESULTS_DIR / "pipeline_comparison.csv", index=False)

# %% [markdown]
# ## 11. Conclusion
#
# **Research question:** Does perception of visual symmetry automatically
# modulate early visual ERP components (P1, N1) and elicit affective responses?
#
# **Our pipeline's answer:**
# [Fill in: did you replicate the P1/N1 modulation? Was it significant?
# In which direction? Does this match the paper?]
#
# **Robustness conclusion:**
# [Fill in: do the findings hold with a different reference and automated
# artefact removal? This is the core contribution of the project.]

# %%
print(f"\nNotebook 03 complete. All results saved to: {RESULTS_DIR}")
print(f"Grand average based on {len(all_evokeds['Regular'])} subjects.")
