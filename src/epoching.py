"""
epoching.py
===========
Functions for cutting epochs from preprocessed continuous data, applying
artefact rejection, and computing evoked (averaged) responses.
"""

import numpy as np
import mne

from src.config import (
    EVENT_ID,
    EPOCH_TMIN, EPOCH_TMAX, BASELINE,
    EPOCH_REJECT,
    MIN_EPOCHS_PER_CONDITION,
)

import mne

def create_epochs(raw):
    """
    Create epochs using stimulus channel (values 1 and 3)
    """

    # --- STEP 1: Extract events from stim channel ---
    events = mne.find_events(raw, shortest_event=1)

    print("DEBUG: Unique event values:", set(events[:, 2]))

    # --- STEP 2: Define mapping (based on your events.tsv/json) ---
    event_id = {
        "Regular": 1,
        "Random": 3
    }

    # --- STEP 3: Sanity check ---
    for name, val in event_id.items():
        count = (events[:, 2] == val).sum()
        print(f"Events found — {name}: {count}")

        if count == 0:
            raise ValueError(f"No events found for condition '{name}' (value={val}).")

    # --- STEP 4: Create epochs ---
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=-0.2,
        tmax=0.8,
        baseline=(None, 0),
        preload=True
    )

    return epochs

def drop_bad_epochs(epochs: mne.Epochs):
    """
    Apply amplitude rejection and return cleaned epochs + log.
    """

    # FIX: drop_bad() works in-place → do NOT assign it
    epochs_clean = epochs.copy()
    epochs_clean.drop_bad(verbose=False)

    rejection_log = {}

    for cond in EVENT_ID:
        n_orig  = len(epochs[cond])
        n_clean = len(epochs_clean[cond])
        n_drop  = n_orig - n_clean
        pct     = 100 * n_clean / n_orig if n_orig > 0 else 0

        rejection_log[cond] = {
            "original": n_orig,
            "kept": n_clean,
            "dropped": n_drop,
            "percent_kept": round(pct, 1),
        }

        status = "OK" if n_clean >= MIN_EPOCHS_PER_CONDITION else "⚠ LOW"
        print(f"  {cond}: {n_clean}/{n_orig} epochs kept ({pct:.1f}%)  [{status}]")

    return epochs_clean, rejection_log


def check_subject_quality(rejection_log: dict, subject: str) -> bool:
    """
    Flag a subject as an outlier if too many epochs were rejected.

    A subject is flagged if either condition drops below MIN_EPOCHS_PER_CONDITION.
    This is a hard exclusion criterion that should be documented in the report.

    Parameters
    ----------
    rejection_log : dict
        Output from drop_bad_epochs().
    subject : str

    Returns
    -------
    is_ok : bool
        True if the subject meets quality criteria.
    """
    for cond, log in rejection_log.items():
        if log["kept"] < MIN_EPOCHS_PER_CONDITION:
            print(
                f"  ⚠ Subject {subject} flagged: only {log['kept']} {cond} epochs "
                f"(threshold: {MIN_EPOCHS_PER_CONDITION})."
            )
            return False
    return True


def compute_evokeds(epochs_clean: mne.Epochs) -> dict[str, mne.Evoked]:
    """
    Average epochs per condition to produce evoked responses.

    Averaging across trials cancels random noise (which averages to zero
    if the noise is uncorrelated with the stimulus). The resulting evoked
    response represents the deterministic brain response time-locked to
    the stimulus.

    We compute evokeds per condition so we can:
    1. Plot condition-specific ERP waveforms.
    2. Compute a difference wave (Regular − Random).
    3. Stack subject evokeds for grand averages.

    Parameters
    ----------
    epochs_clean : mne.Epochs

    Returns
    -------
    evokeds : dict[str, mne.Evoked]
        Keys match EVENT_ID (e.g. 'Regular', 'Random').
    """
    evokeds = {}
    for cond in EVENT_ID:
        evokeds[cond] = epochs_clean[cond].average()
        evokeds[cond].comment = cond
    return evokeds


def compute_difference_wave(evokeds: dict[str, mne.Evoked]) -> mne.Evoked:
    """
    Compute the difference wave: Regular minus Random.

    The difference wave isolates the neural response that is specific to
    symmetric (Regular) patterns. Positive values indicate greater
    positivity for Regular; negative values indicate greater negativity
    (e.g., P1 enhancement or N1 enhancement for symmetry).

    In the original paper, the SPN is a sustained negativity for Regular
    vs Random (300–1000 ms). If our pipeline is robust, we expect a
    similar pattern in this difference wave at posterior channels.

    Parameters
    ----------
    evokeds : dict[str, mne.Evoked]

    Returns
    -------
    diff_wave : mne.Evoked
    """
    diff_wave = mne.combine_evoked(
        [evokeds["Regular"], evokeds["Random"]],
        weights=[1, -1],
    )
    diff_wave.comment = "Regular - Random"
    return diff_wave


def run_epoching_pipeline(raw: mne.io.BaseRaw, subject: str) -> tuple:
    """
    High-level function: run full epoching pipeline for one subject.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Preprocessed raw data.
    subject : str

    Returns
    -------
    epochs_clean : mne.Epochs
    evokeds : dict[str, mne.Evoked]
    diff_wave : mne.Evoked
    rejection_log : dict
    is_ok : bool
    """
    print(f"\n  Epoching subject {subject}...")
    epochs = create_epochs(raw)
    epochs_clean, rejection_log = drop_bad_epochs(epochs)
    is_ok = check_subject_quality(rejection_log, subject)
    evokeds = compute_evokeds(epochs_clean)
    diff_wave = compute_difference_wave(evokeds)
    return epochs_clean, evokeds, diff_wave, rejection_log, is_ok
