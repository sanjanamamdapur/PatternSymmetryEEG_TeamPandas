"""
plotting.py
===========
All visualisation functions for the EEG project.

Each function saves a figure to RESULTS_DIR and also returns it so
notebooks can display it inline.

Design note: all plotting functions include a 'title' parameter so that
when figures are saved, the filename is descriptive and traceable.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mne

from src.config import (
    RESULTS_DIR, CONDITION_COLORS, FIGURE_DPI,
    ROI_CHANNELS, P1_WINDOW_MS, N1_WINDOW_MS, SPN_WINDOW_MS,
    EVENT_ID,
)

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "figure.dpi": FIGURE_DPI,
})


# ── Raw data overview ─────────────────────────────────────────────────────────

def plot_raw_overview(raw: mne.io.BaseRaw, subject: str, duration: float = 10.0):
    """
    Plot a short segment of the raw signal as a first sanity check.

    This is the 'first impression' of the data required for Milestone 2.
    We look for:
    - Obvious bad channels (flat lines, very high amplitude).
    - 50 Hz line noise (visible as regular oscillations before filtering).
    - Eye blink artefacts in EOG channels.

    Parameters
    ----------
    raw : mne.io.BaseRaw
    subject : str
    duration : float
        How many seconds of data to show.
    """
    fig = raw.plot(
        duration=duration,
        n_channels=30,
        scalings={"eeg": 100e-6, "eog": 200e-6},
        title=f"Raw signal — sub-{subject} (first {duration:.0f} s)",
        show=False,
    )
    fig.savefig(RESULTS_DIR / f"sub-{subject}_raw_overview.png", bbox_inches="tight")
    return fig


def plot_power_spectrum(raw: mne.io.BaseRaw, subject: str):
    """
    Plot the power spectral density (PSD) of the raw signal.

    This sanity check lets us verify:
    - The 50 Hz notch filter worked (dip at 50 Hz after filtering).
    - The 1/f characteristic of EEG is preserved (power decreases with frequency).
    - No unexpected peaks (e.g., artefacts from the recording setup).

    Parameters
    ----------
    raw : mne.io.BaseRaw
    subject : str
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    spectrum = raw.compute_psd(picks="eeg", fmax=80, verbose=False)
    spectrum.plot(axes=ax, show=False)
    ax.set_title(f"Power spectral density — sub-{subject}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (dB)")
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"sub-{subject}_psd.png", bbox_inches="tight")
    return fig


# ── ICA ──────────────────────────────────────────────────────────────────────

def plot_ica_components(
    ica: mne.preprocessing.ICA,
    raw: mne.io.BaseRaw,
    exclude_indices: list[int],
    subject: str,
    n_components: int = 20,
):
    """
    Plot ICA component topographies and time courses.

    We show the first n_components, highlighting the ones selected for removal
    in red. This allows visual verification that we are removing genuine
    artefacts (e.g., the frontal topography typical of eye blinks) and not
    brain signal.

    This plot must be included in the report with a one-line justification
    for each removed component.
    """
    fig = ica.plot_components(
        picks=range(n_components),
        title=f"ICA components — sub-{subject} | Red = excluded",
        show=False,
    )
    for ax_idx, ax in enumerate(fig.axes):
        if ax_idx in exclude_indices:
            ax.set_facecolor("#fee0e0")
            ax.set_title(ax.get_title(), color="red")
    fig.savefig(RESULTS_DIR / f"sub-{subject}_ica_components.png", bbox_inches="tight")
    return fig


def plot_ica_overlay(
    raw: mne.io.BaseRaw,
    ica: mne.preprocessing.ICA,
    subject: str,
):
    """
    Plot raw signal before and after ICA removal at ROI channels.

    The overlay shows the artefact waveform (difference between original
    and cleaned signal). This is a crucial sanity check: the removed
    signal should look like eye blinks/muscle noise, not brain activity.
    """
    fig = ica.plot_overlay(
        raw,
        exclude=ica.exclude,
        picks="eeg",
        title=f"ICA overlay — sub-{subject} (grey=original, red=removed)",
        show=False,
    )
    fig.savefig(RESULTS_DIR / f"sub-{subject}_ica_overlay.png", bbox_inches="tight")
    return fig


# ── Epochs ───────────────────────────────────────────────────────────────────

def plot_epoch_rejection_summary(rejection_summary_df, outlier_subjects: list[str]):
    """
    Bar chart showing % epochs kept per subject and condition.

    This is one of the key sanity check visualisations. We expect most
    subjects to retain >80% of epochs. Subjects below the MIN_EPOCHS
    threshold are highlighted.

    Parameters
    ----------
    rejection_summary_df : pd.DataFrame
        Output of build_rejection_summary().
    outlier_subjects : list[str]
        Subject IDs flagged as outliers.
    """
    df = rejection_summary_df
    subjects = df.index.tolist()
    x = np.arange(len(subjects))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 5))

    bars1 = ax.bar(
        x - width / 2,
        df["Regular_pct_kept"],
        width,
        label="Regular",
        color=CONDITION_COLORS["Regular"],
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        df["Random_pct_kept"],
        width,
        label="Random",
        color=CONDITION_COLORS["Random"],
        alpha=0.8,
    )

    # Mark outlier subjects with a red border.
    for i, subj in enumerate(subjects):
        if subj in outlier_subjects:
            for bar in [bars1[i], bars2[i]]:
                bar.set_edgecolor("red")
                bar.set_linewidth(2)

    ax.axhline(y=62.5, color="gray", linestyle="--", linewidth=1,
               label="Min threshold (50/80 = 62.5%)")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45, ha="right")
    ax.set_ylabel("Epochs kept (%)")
    ax.set_xlabel("Subject")
    ax.set_title("Epoch rejection summary (red border = outlier subject)")
    ax.legend()
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "epoch_rejection_summary.png", bbox_inches="tight")
    return fig


# ── ERP waveforms ─────────────────────────────────────────────────────────────

def plot_erp_waveforms(
    grand_averages: dict[str, mne.Evoked],
    subject: str = "grand",
    channels: list[str] = None,
):
    """
    Plot ERP waveforms for Regular and Random conditions at ROI channels.

    This replicates Figure 1 from the original paper (ERP waveforms at
    posterior channels). We expect:
    - P1 (~100 ms): positive peak, potentially larger for Regular.
    - N1 (~170 ms): negative peak, potentially more negative for Regular.

    The shaded bands mark the P1 and N1 analysis windows.

    Parameters
    ----------
    grand_averages : dict[str, mne.Evoked]
        Grand average or single-subject evokeds.
    subject : str
        'grand' for group plot, or subject ID for single-subject.
    channels : list[str]
        Defaults to ROI_CHANNELS (PO7, PO8).
    """
    if channels is None:
        channels = ROI_CHANNELS

    fig, ax = plt.subplots(figsize=(10, 5))
    times_ms = grand_averages["Regular"].times * 1000

    for cond, evoked in grand_averages.items():
        # Average across ROI channels.
        ch_idx = [evoked.ch_names.index(ch) for ch in channels if ch in evoked.ch_names]
        if not ch_idx:
            print(f"  Warning: none of {channels} found in evoked. Skipping.")
            continue
        amplitude_uV = evoked.data[ch_idx, :].mean(axis=0) * 1e6
        ax.plot(times_ms, amplitude_uV, label=cond, color=CONDITION_COLORS[cond], linewidth=2)

    # Mark analysis windows.
    for window, label, color in [
        (P1_WINDOW_MS, "P1", "orange"),
        (N1_WINDOW_MS, "N1", "steelblue"),
    ]:
        ax.axvspan(window[0], window[1], alpha=0.15, color=color, label=f"{label} window")

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", label="Stimulus onset")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (µV)")
    ax.set_title(f"ERP waveforms at {', '.join(channels)} — {subject}")
    ax.legend(loc="upper right", fontsize=9)
    ax.invert_yaxis()  # ERP convention: negative up
    ax.set_xlim(-200, 1000)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"erp_waveforms_{subject}.png", bbox_inches="tight")
    return fig


def plot_difference_wave(
    diff_wave: mne.Evoked,
    subject: str = "grand",
    channels: list[str] = None,
):
    """
    Plot the difference wave (Regular − Random) with a zero line.

    Positive values = Regular more positive than Random.
    Negative values = Regular more negative than Random (e.g. SPN).

    If the paper's findings are robust to our pipeline, we expect:
    - A positive deflection around 80–130 ms (P1 enhancement for symmetry).
    - A negative deflection around 150–200 ms (N1 enhancement for symmetry).
    - A sustained negative shift from 300–1000 ms (SPN).
    """
    if channels is None:
        channels = ROI_CHANNELS

    fig, ax = plt.subplots(figsize=(10, 4))
    times_ms = diff_wave.times * 1000

    ch_idx = [diff_wave.ch_names.index(ch) for ch in channels if ch in diff_wave.ch_names]
    diff_uV = diff_wave.data[ch_idx, :].mean(axis=0) * 1e6

    ax.plot(times_ms, diff_uV, color="black", linewidth=2, label="Regular − Random")
    ax.fill_between(times_ms, diff_uV, 0, where=(diff_uV < 0), color="#378ADD", alpha=0.2, label="Negativity")
    ax.fill_between(times_ms, diff_uV, 0, where=(diff_uV > 0), color="#E24B4A", alpha=0.2, label="Positivity")

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

    # Mark SPN window.
    ax.axvspan(*SPN_WINDOW_MS, alpha=0.08, color="purple", label="SPN window")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude difference (µV)")
    ax.set_title(f"Difference wave (Regular − Random) at {', '.join(channels)} — {subject}")
    ax.legend(fontsize=9)
    ax.set_xlim(-200, 1000)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"difference_wave_{subject}.png", bbox_inches="tight")
    return fig


# ── Topographic maps ──────────────────────────────────────────────────────────

def plot_topomap_series(
    grand_averages: dict[str, mne.Evoked],
    condition: str = "Regular",
    time_windows: list[tuple] = None,
):
    """
    Plot topographic maps of scalp distribution across time windows.

    This replicates the MATLAB Toposubplotter_2020.m output.
    Topomaps show WHERE on the scalp the effect is localised. For P1/N1,
    we expect the effect to be maximal at posterior (occipital-parietal)
    electrodes, consistent with primary visual cortex processing.

    Parameters
    ----------
    grand_averages : dict[str, mne.Evoked]
    condition : str
    time_windows : list of (tmin_ms, tmax_ms) tuples
        Defaults to every 100 ms from −200 to 1000 ms.
    """
    if time_windows is None:
        time_windows = [(t, t + 100) for t in range(-200, 800, 100)]

    evoked = grand_averages[condition]
    times_s = [np.mean(w) / 1000.0 for w in time_windows]

    fig, axes = plt.subplots(2, 6, figsize=(14, 5))
    axes = axes.flatten()

    for ax, t_s, (tmin, tmax) in zip(axes, times_s, time_windows):
        evoked.plot_topomap(
            times=t_s,
            axes=ax,
            show=False,
            colorbar=False,
            time_format="",
            vlim=(-3e-6, 3e-6),
        )
        ax.set_title(f"{tmin}–{tmax} ms", fontsize=9)

    fig.suptitle(f"Topographic maps — {condition}", fontsize=12)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"topomap_{condition}.png", bbox_inches="tight")
    return fig


def plot_difference_topomap(grand_averages: dict[str, mne.Evoked]):
    """
    Plot topographic maps of the Regular − Random difference.

    If the P1/N1 effect is real, the difference should show a
    posterior distribution in the P1 and N1 time windows, not a
    frontal or uniform distribution (which would suggest an artefact).
    """
    diff = mne.combine_evoked(
        [grand_averages["Regular"], grand_averages["Random"]],
        weights=[1, -1],
    )
    time_windows = [(80, 130), (150, 200), (300, 800)]
    times_s = [np.mean(w) / 1000.0 for w in time_windows]
    labels = ["P1 window\n(80–130 ms)", "N1 window\n(150–200 ms)", "SPN window\n(300–1000 ms)"]

    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, t_s, lbl in zip(axes, times_s, labels):
        diff.plot_topomap(
            times=t_s,
            axes=ax,
            show=False,
            colorbar=False,
            time_format="",
            vlim=(-2e-6, 2e-6),
        )
        ax.set_title(lbl, fontsize=10)

    # Add a single shared colorbar
    im = axes[0].collections[0]
    cbar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02)
    cbar.set_label("Amplitude (V)")

    fig.suptitle("Topographic maps — Regular minus Random difference", fontsize=12)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "topomap_difference.png", bbox_inches="tight")
    return fig


# ── Statistics ───────────────────────────────────────────────────────────────

def plot_amplitude_distributions(metrics_df, component: str = "P1"):
    """
    Paired violin + strip plot showing per-subject amplitudes for Regular vs Random.

    This makes the within-subject comparison visible: each dot is one subject,
    connected lines show the direction of the effect. If the paper's finding
    is replicated, we expect most lines to go in the same direction.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    reg_col = f"Regular_{component}"
    ran_col = f"Random_{component}"

    # Draw connecting lines (one per subject).
    for _, row in metrics_df.iterrows():
        ax.plot([0, 1], [row[reg_col], row[ran_col]], color="gray", alpha=0.4, linewidth=1)

    # Draw condition means as large dots.
    for x, col, cond in [(0, reg_col, "Regular"), (1, ran_col, "Random")]:
        vals = metrics_df[col].values
        ax.scatter(
            [x] * len(vals), vals,
            color=CONDITION_COLORS[cond], alpha=0.7, s=40, zorder=3,
        )
        ax.plot(x, vals.mean(), marker="D", markersize=10,
                color=CONDITION_COLORS[cond], zorder=4, label=f"{cond} mean")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Regular", "Random"])
    ax.set_ylabel("Mean amplitude (µV)")
    ax.set_title(f"{component} amplitude — Regular vs Random at {', '.join(ROI_CHANNELS)}")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / f"amplitude_distribution_{component}.png", bbox_inches="tight")
    return fig


def plot_stats_table(stats_df):
    """
    Render the statistical results table as a matplotlib figure for the report.
    """
    display_cols = ["T", "dof", "p-val", "cohen-d", "mean_diff_uV", "sd_diff_uV"]
    available = [c for c in display_cols if c in stats_df.columns]
    df_disp = stats_df[available].round(3)

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=df_disp.values,
        rowLabels=df_disp.index,
        colLabels=df_disp.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    ax.set_title("Paired t-tests: Regular vs Random (PO7/PO8)", fontsize=11, pad=12)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "stats_table.png", bbox_inches="tight")
    return fig
