"""
analysis.py
===========
Functions for extracting ERP metrics (P1/N1 mean amplitudes) and
running group-level statistics.
"""

import numpy as np
import pandas as pd
import mne
import pingouin as pg

from src.config import (
    ROI_CHANNELS,
    P1_WINDOW_MS, N1_WINDOW_MS, SPN_WINDOW_MS,
    EVENT_ID,
)


# ── Grand averages ───────────────────────────────────────────────────────────

def compute_grand_averages(
    all_evokeds: dict[str, list[mne.Evoked]]
) -> dict[str, mne.Evoked]:
    """
    Compute grand-average evoked responses across subjects.

    The grand average is the mean of the individual subject ERPs.
    This is the standard way to show group-level ERP results because
    individual subject noise averages out further across subjects.

    We weight all subjects equally (each subject contributes one ERP).
    This is equivalent to a fixed-effects approach — each ERP is already
    the average over that subject's trials.

    Parameters
    ----------
    all_evokeds : dict[str, list[mne.Evoked]]
        Keys are condition names; values are lists of one Evoked per subject.

    Returns
    -------
    grand_averages : dict[str, mne.Evoked]
    """
    grand_averages = {}
    for cond, evoked_list in all_evokeds.items():
        grand_averages[cond] = mne.grand_average(evoked_list)
        grand_averages[cond].comment = cond
        print(f"  Grand average '{cond}': {len(evoked_list)} subjects")
    return grand_averages


# ── ERP amplitude extraction ─────────────────────────────────────────────────

def extract_mean_amplitude(
    evoked: mne.Evoked,
    tmin_ms: float,
    tmax_ms: float,
    channels: list[str] = None,
) -> float:
    """
    Extract mean amplitude in a time window averaged across ROI channels.

    Mean amplitude (rather than peak amplitude) is used because:
    1. It is more stable across subjects (less sensitive to single-trial noise).
    2. It is the standard measure for broad ERP components like P1/N1.
    3. It matches the approach used in the MATLAB scripts from the original paper
       (Thin_SPN_extractor_2020.m uses mean over a time window).

    Parameters
    ----------
    evoked : mne.Evoked
    tmin_ms, tmax_ms : float
        Time window in milliseconds.
    channels : list[str]
        Channels to average. Defaults to ROI_CHANNELS from config.

    Returns
    -------
    amplitude : float
        Mean amplitude in µV.
    """
    if channels is None:
        channels = ROI_CHANNELS

    tmin_s = tmin_ms / 1000.0
    tmax_s = tmax_ms / 1000.0

    # Crop to window and pick ROI channels.
    evoked_cropped = evoked.copy().crop(tmin=tmin_s, tmax=tmax_s)
    evoked_roi = evoked_cropped.pick_channels(channels)

    # Mean over time and channels, convert from Volts to µV.
    amplitude = evoked_roi.data.mean() * 1e6
    return amplitude


def extract_subject_metrics(
    evokeds: dict[str, mne.Evoked],
    subject: str,
) -> dict:
    """
    Extract P1, N1 (and optionally SPN) amplitudes for one subject.

    Parameters
    ----------
    evokeds : dict[str, mne.Evoked]
    subject : str

    Returns
    -------
    metrics : dict
        Flat dict with keys like 'Regular_P1', 'Random_N1', etc.
    """
    metrics = {"subject": subject}
    for cond in EVENT_ID:
        ev = evokeds[cond]
        metrics[f"{cond}_P1"] = extract_mean_amplitude(ev, *P1_WINDOW_MS)
        metrics[f"{cond}_N1"] = extract_mean_amplitude(ev, *N1_WINDOW_MS)
        metrics[f"{cond}_SPN"] = extract_mean_amplitude(ev, *SPN_WINDOW_MS)
    return metrics


def build_metrics_dataframe(all_metrics: list[dict]) -> pd.DataFrame:
    """
    Combine per-subject metric dicts into a tidy DataFrame.

    Parameters
    ----------
    all_metrics : list[dict]
        One dict per subject (output of extract_subject_metrics).

    Returns
    -------
    df : pd.DataFrame
        Columns: subject, Regular_P1, Regular_N1, Regular_SPN,
                         Random_P1, Random_N1, Random_SPN
    """
    df = pd.DataFrame(all_metrics)
    df = df.set_index("subject")
    return df


# ── Statistics ───────────────────────────────────────────────────────────────

def run_paired_ttest(
    df: pd.DataFrame,
    component: str,
) -> pd.DataFrame:
    """
    Run a paired t-test comparing Regular vs Random for one ERP component.

    We use a paired t-test because each subject contributes one data point
    per condition (the mean amplitude), and conditions are fully within-subject
    (each participant saw both Regular and Random stimuli).

    Effect size: Cohen's d, computed by pingouin. This lets us compare the
    magnitude of the effect to the original paper's results.

    Interpretation: A significant difference in P1 amplitude would support
    the paper's finding that symmetry modulates early visual processing.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_metrics_dataframe().
    component : str
        One of 'P1', 'N1', 'SPN'.

    Returns
    -------
    result : pd.DataFrame
        Pingouin t-test output including t, df, p-value, Cohen's d, CI.
    """
    regular_col = f"Regular_{component}"
    random_col  = f"Random_{component}"

    result = pg.ttest(
        df[regular_col],
        df[random_col],
        paired=True,
        alternative="two-sided",
    )
    result.index = [component]
    return result


def run_all_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run paired t-tests for P1, N1, and SPN and combine into one table.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    stats_table : pd.DataFrame
    """
    results = []
    for comp in ["P1", "N1", "SPN"]:
        res = run_paired_ttest(df, comp)
        results.append(res)

    stats_table = pd.concat(results)

    # Add mean difference and SD for reporting.
    for comp in ["P1", "N1", "SPN"]:
        diff = df[f"Regular_{comp}"] - df[f"Random_{comp}"]
        stats_table.loc[comp, "mean_diff_uV"] = diff.mean()
        stats_table.loc[comp, "sd_diff_uV"]   = diff.std()

    return stats_table


# ── Rejection summary ─────────────────────────────────────────────────────────

def build_rejection_summary(all_rejection_logs: dict) -> pd.DataFrame:
    """
    Build a summary DataFrame of epoch rejection rates per subject and condition.

    This is a key sanity check visualisation. If one subject has >50% rejection
    in any condition, it warrants inspection of the raw data and potentially
    exclusion.

    Parameters
    ----------
    all_rejection_logs : dict
        Keys are subject IDs; values are rejection_log dicts from drop_bad_epochs().

    Returns
    -------
    df : pd.DataFrame
        Index = subjects, columns = conditions with percent_kept values.
    """
    rows = []
    for subj, log in all_rejection_logs.items():
        row = {"subject": subj}
        for cond, info in log.items():
            row[f"{cond}_pct_kept"] = info["percent_kept"]
            row[f"{cond}_n_kept"]   = info["kept"]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("subject")
    return df
