"""
preprocessing.py
================
All preprocessing steps from raw BIDS loading through ICA.

Design principle: every function takes a Raw/ICA object and returns a
modified copy (or new object) so steps are composable and individually
testable. No side-effects on the input object.

Each parameter choice is documented inline.

Dataset-specific bug fixes (documented for the report):
  1. Channel names in channels.tsv are quote-wrapped (e.g. "'PO7'") --
     stripped with .strip("'") before any renaming or montage matching.
  2. EXG5/6 become M1/M2 (mastoids) and EXG7/8 are spare channels --
     all four must be set to type "misc" so pick("eeg") excludes them,
     otherwise ICLabel crashes with "Channel position missing".
  3. ICLabel requires EEG-only data filtered 1-100 Hz with average reference.
     run_ica() builds a dedicated raw_ica copy that satisfies all three
     requirements and returns it alongside the fitted ICA object.
"""

import numpy as np
import mne
from mne_bids import BIDSPath, read_raw_bids
from mne_icalabel import label_components

from src.config import (
    BIDS_ROOT, TASK,
    EOG_CHANNEL_RENAME, EOG_CHANNEL_NAMES,
    HIGHPASS_HZ, LOWPASS_HZ, NOTCH_HZ,
    ICA_N_COMPONENTS, ICA_RANDOM_STATE, ICLABEL_THRESHOLD,
    EVENT_ID,
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def make_bids_path(subject: str) -> BIDSPath:
    """Construct the BIDSPath for a given subject, e.g. '005'."""
    return BIDSPath(subject=subject, task=TASK, datatype="eeg", root=BIDS_ROOT)


def load_raw(subject: str, preload: bool = True) -> mne.io.BaseRaw:
    """
    Load raw BIDS data for one subject and apply all channel fixes.

    Fixes applied in order
    ----------------------
    Fix 1 -- strip quote characters from channel names.
        channels.tsv wraps every name in single quotes, e.g. "'PO7'".
        MNE reads these verbatim. Without stripping, montage matching,
        EOG renaming, and ROI lookups all silently fail.

    Fix 2 -- rename EXG channels and assign correct channel types.
        EXG1-4 -> EOG (eye signals, used by ICA and find_bads_eog).
        EXG5-6 -> M1/M2 (mastoids), set to "misc".
        EXG7-8 -> spare, set to "misc".
        CRITICAL: M1, M2, EXG7, EXG8 must be "misc" not "eeg".
        When they stay as EEG, pick("eeg") inside run_ica() keeps them,
        and ICLabel crashes with "Channel position for M1/M2/... missing".

    Fix 3 -- set biosemi64 montage.
        After quote-stripping, EEG names match the built-in montage.
        The four misc channels have no entry -> on_missing='warn' is safe.

    Note on onset units
    -------------------
    events.json declares onset units as "ms" but values like 12.87, 18.37
    are clearly seconds (recording is 901 s long). mne-bids follows the
    BIDS spec and treats them as seconds regardless. We verify this below.
    """
    bids_path = make_bids_path(subject)
    raw = read_raw_bids(bids_path, verbose=False)

    if preload:
        raw.load_data()

    # Fix 1: strip quote wrapping
    raw.rename_channels(lambda ch: ch.strip("'"))

    # Fix 2: rename and retype EXG channels
    raw.rename_channels(EOG_CHANNEL_RENAME)
    raw.set_channel_types({ch: "eog" for ch in EOG_CHANNEL_NAMES})
    raw.set_channel_types({
        "M1":   "misc",   # left mastoid -- no 10-20 montage position
        "M2":   "misc",   # right mastoid -- no 10-20 montage position
        "EXG7": "misc",   # spare channel
        "EXG8": "misc",   # spare channel
    })

    # Fix 3: montage
    montage = mne.channels.make_standard_montage("biosemi64")
    raw.set_montage(montage, on_missing="warn")

    # Sanity check: onset values must be in seconds, not milliseconds
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_ID, verbose=False)
    if len(events) > 0:
        max_onset_s = events[:, 0].max() / raw.info["sfreq"]
        assert max_onset_s < raw.times[-1], (
            f"sub-{subject}: max event onset ({max_onset_s:.1f} s) exceeds "
            f"recording duration ({raw.times[-1]:.1f} s). "
            "Possible onset unit mismatch (ms vs s)."
        )

    return raw


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Bandpass 0.1-40 Hz + notch at 50/100 Hz.

    Rationale
    ---------
    0.1 Hz high-pass: removes DC drift without distorting P1/N1 or the
        slower SPN component (300-1000 ms). A 1 Hz cutoff would distort
        the SPN; 0.1 Hz is the safe compromise.
    40 Hz low-pass: P1/N1 are below 15 Hz. 40 Hz removes muscle noise
        while staying well above the signal of interest.
    50/100 Hz notch: UK mains frequency and first harmonic, confirmed
        from eeg.json ('PowerLineFrequency': 50).
    FIR (firwin): linear phase response preserves ERP peak latency.
    """
    raw_filt = raw.copy()
    raw_filt.filter(
        l_freq=HIGHPASS_HZ, h_freq=LOWPASS_HZ,
        picks=["eeg", "eog"],
        method="fir", fir_design="firwin",
        verbose=False,
    )
    raw_filt.notch_filter(
        freqs=[NOTCH_HZ, NOTCH_HZ * 2],
        picks=["eeg", "eog"],
        verbose=False,
    )
    return raw_filt


# ---------------------------------------------------------------------------
# Bad channels
# ---------------------------------------------------------------------------

def find_bad_channels(raw: mne.io.BaseRaw) -> list:
    """
    Flag flat channels (SD < 1% of median) and noisy channels (SD > 5x median).

    Automated thresholding applies the same objective criterion to all
    subjects, making the exclusion rule documentable and reproducible.
    """
    eeg_idx = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks="eeg")
    std = data.std(axis=1)
    med = np.median(std)
    names = [raw.ch_names[i] for i in eeg_idx]
    return [ch for ch, s in zip(names, std) if s < med * 0.01 or s > med * 5.0]


def interpolate_bad_channels(raw: mne.io.BaseRaw, bad_channels: list) -> mne.io.BaseRaw:
    """
    Spherical-spline interpolation of flagged channels (Perrin et al., 1989).

    Interpolation keeps the full 64-channel layout intact for topographic
    plots and average referencing -- both would be biased by dropped channels.
    """
    raw_interp = raw.copy()
    raw_interp.info["bads"] = bad_channels
    if bad_channels:
        raw_interp.interpolate_bads(reset_bads=True, verbose=False)
        print(f"  Interpolated {len(bad_channels)} channel(s): {bad_channels}")
    else:
        print("  No bad channels detected.")
    return raw_interp


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def set_average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """
    Re-reference to the common average of all 64 EEG channels.

    Applied AFTER ICA because average reference reduces data rank by 1,
    which would cause ICA to fit one fewer component than requested.
    Average reference is preferred over the authors' likely linked-mastoid
    reference because it makes no assumption about any electrode being
    electrically neutral -- a deliberate pipeline difference to test
    robustness.
    """
    raw_ref = raw.copy()
    raw_ref.set_eeg_reference("average", projection=False, verbose=False)
    return raw_ref


# ---------------------------------------------------------------------------
# ICA
# ---------------------------------------------------------------------------

def run_ica(raw_original: mne.io.BaseRaw) -> tuple:
    """
    Fit ICA and prepare the raw copy required by ICLabel.

    Parameters
    ----------
    raw_original : mne.io.BaseRaw
        The UNFILTERED loaded raw (output of load_raw, before filter_raw).
        This is required because ICLabel needs genuine 1-100 Hz data.
        If the already-lowpassed (0.1-40 Hz) data were passed instead,
        the h_freq=100 filter call below would have no effect above 40 Hz,
        ICLabel would see only 1-40 Hz data, issue warnings, and return
        a malformed y_pred_proba array causing an IndexError.

    Returns
    -------
    ica : mne.preprocessing.ICA
        Fitted ICA object (NOT yet applied to the main data stream).
    raw_ica : mne.io.BaseRaw
        EEG-only, 1-100 Hz, average-referenced copy used by ICLabel.

    ICLabel requirements (all three must be satisfied):
        a) EEG channels only -- no EOG, misc, or stim.
        b) Bandpass filtered 1-100 Hz (genuine, starting from raw data).
        c) Average reference applied.

    The ICA solution fitted here is later applied to raw_filtered
    (0.1-40 Hz) via apply_ica(), so slow ERP components are preserved.

    Algorithm: extended Infomax (method='infomax', extended=True).
    ICLabel was trained exclusively on Infomax decompositions. FastICA
    triggers a warning and degrades classification accuracy.
    """
    # Build ICLabel-compatible copy from the ORIGINAL (unfiltered) raw.
    # Starting from raw_original means the 1-100 Hz bandpass is genuine --
    # no pre-existing lowpass cap at 40 Hz limits the upper frequency.
    raw_ica = raw_original.copy().pick("eeg")
    raw_ica.filter(l_freq=1.0, h_freq=100.0, verbose=False)
    raw_ica.set_eeg_reference("average", projection=False, verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=ICA_N_COMPONENTS,
        method="infomax",
        fit_params={"extended": True},  # required for ICLabel
        random_state=ICA_RANDOM_STATE,  # reproducibility
        max_iter=1000,
        verbose=False,
    )
    ica.fit(raw_ica, verbose=False)

    # Return BOTH: ica (apply to raw_filtered) and raw_ica (pass to ICLabel)
    return ica, raw_ica


def label_ica_components(raw_ica: mne.io.BaseRaw,
                          ica: mne.preprocessing.ICA) -> dict:
    """
    Classify ICA components using ICLabel.

    Passes raw_ica (EEG-only, 1-100 Hz, avg-ref) as required.
    Returns None on failure so the pipeline can fall back gracefully
    to EOG-correlation-only selection.

    ICLabel assigns probabilities to 7 classes:
        Brain(0), Muscle(1), Eye(2), Heart(3),
        Line Noise(4), Channel Noise(5), Other(6).
    """
    try:
        return label_components(raw_ica, ica, method="iclabel")
    except Exception as e:
        print(f"  WARNING: ICLabel failed ({type(e).__name__}: {e})")
        print("  Falling back to EOG-correlation-only selection.")
        return None


def select_artefact_components(ica: mne.preprocessing.ICA,
                                ic_labels,
                                raw_with_eog: mne.io.BaseRaw) -> list:
    """
    Choose which ICA components to remove.

    Two-stage strategy
    ------------------
    Stage 1 -- ICLabel (when available):
        Remove components where P(muscle) > threshold OR P(eye) > threshold.
        Threshold from config (default 0.8) is conservative -- keeps
        borderline components to minimise accidental removal of brain signal.

    Stage 2 -- EOG correlation (always run as a cross-check):
        Remove components with Pearson r > 0.7 against any EOG channel.
        Catches eye components ICLabel may miss, especially in subjects
        with infrequent blinks.

    Heart and Line Noise are NOT auto-removed: false-positive rates are
    higher for these classes and the cost of removing brain signal outweighs
    the benefit.

    raw_with_eog must be raw_filtered (0.1-40 Hz) which still has the four
    EOG channels -- raw_ica dropped them with pick("eeg").
    """
    exclude = set()

    # Stage 1: ICLabel
    if ic_labels is not None:
        labels = ic_labels["labels"]
        probs  = ic_labels["y_pred_proba"]
        for i, (lbl, p) in enumerate(zip(labels, probs)):
            # y_pred_proba is (n_components, 7) in correct ICLabel output.
            # Guard against degenerate 1D output (happens when ICLabel
            # receives out-of-spec data) so we fail safe instead of crash.
            if p.ndim == 0 or (hasattr(p, "__len__") and len(p) < 7):
                continue
            if lbl in ("muscle", "eye") and max(p[1], p[2]) > ICLABEL_THRESHOLD:
                exclude.add(i)
        print(f"  ICLabel: {len(exclude)} component(s) flagged.")
    else:
        print("  ICLabel unavailable -- EOG correlation only.")

    # Stage 2: EOG correlation
    eog_inds, _ = ica.find_bads_eog(
        raw_with_eog,
        ch_name=EOG_CHANNEL_NAMES,
        threshold=0.9,
        verbose=False,
    )
    added = set(eog_inds) - exclude
    if added:
        print(f"  EOG correlation added {len(added)} extra component(s): {sorted(added)}")
    exclude.update(eog_inds)
    # Safety cap: never remove more than 20% of components
    max_remove = int(0.2 * ica.n_components_)
    exclude = set(list(exclude)[:max_remove])
    return sorted(exclude)


def apply_ica(raw: mne.io.BaseRaw,
              ica: mne.preprocessing.ICA,
              exclude_indices: list) -> mne.io.BaseRaw:
    """
    Subtract selected ICA components from the continuous signal.

    Zeroes out the chosen components in ICA space, then back-projects to
    channel space. Equivalent to subtracting the reconstructed artefact
    waveform from each EEG channel.
    """
    ica.exclude = exclude_indices
    raw_clean = raw.copy()
    ica.apply(raw_clean, verbose=False)
    print(f"  ICA applied: {len(exclude_indices)} component(s) removed: {exclude_indices}")
    return raw_clean


# ---------------------------------------------------------------------------
# High-level pipeline
# ---------------------------------------------------------------------------

def preprocess_subject(subject: str, save_ica: bool = False) -> tuple:
    """
    Full preprocessing pipeline for one subject.

    Order: load -> filter -> ICA fit -> ICA label -> ICA apply ->
           bad channel detect -> interpolate -> average reference

    Average reference is applied last to avoid rank-reduction during ICA
    and to ensure spherical-spline interpolation uses a consistent
    channel neighbourhood.

    Returns
    -------
    raw_preprocessed : mne.io.BaseRaw
    ica : mne.preprocessing.ICA
    exclude_indices : list[int]
    bad_channels : list[str]
    """
    from src.config import RESULTS_DIR

    print(f"\n{'='*52}")
    print(f"  Preprocessing sub-{subject}")
    print(f"{'='*52}")

    print("  [1/6] Loading...")
    raw = load_raw(subject)
    n_eeg = len(mne.pick_types(raw.info, eeg=True))
    n_eog = len(mne.pick_types(raw.info, eog=True))
    print(f"        {raw.times[-1]:.1f} s | {n_eeg} EEG, {n_eog} EOG channels")

    print("  [2/6] Filtering (0.1-40 Hz + 50 Hz notch)...")
    raw_filt = filter_raw(raw)

    print(f"  [3/6] Fitting ICA ({ICA_N_COMPONENTS} components)...")
    # IMPORTANT: pass raw (unfiltered) not raw_filt -- run_ica needs the full spectrum up to 100 Hz for ICLabel.
    ica, raw_ica = run_ica(raw)              # returns (ica, raw_ica)

    print("  [4/6] ICLabel + EOG correlation...")
    ic_labels = label_ica_components(raw_ica, ica)
    exclude_indices = select_artefact_components(ica, ic_labels, raw_filt)
    raw_clean = apply_ica(raw_filt, ica, exclude_indices)

    print("  [5/6] Bad channels...")
    bad_channels = find_bad_channels(raw_clean)
    raw_interp = interpolate_bad_channels(raw_clean, bad_channels)

    print("  [6/6] Average reference...")
    raw_preprocessed = set_average_reference(raw_interp)

    if save_ica:
        ica_path = RESULTS_DIR / f"sub-{subject}_ica.fif"
        ica.save(ica_path, overwrite=True)
        print(f"        ICA saved -> {ica_path.name}")

    print(f"\n  Done. Bad channels: {bad_channels or 'none'} | "
          f"ICA removed: {exclude_indices or 'none'}")
    return raw_preprocessed, ica, exclude_indices, bad_channels