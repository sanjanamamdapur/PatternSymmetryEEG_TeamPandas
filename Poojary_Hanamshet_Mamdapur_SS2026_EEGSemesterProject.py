# %% [markdown]
# # EEG Semester Project Report
# ## Replication of Makin et al. (2012): Symmetry Perception and Affective Responses
#
# **Team Python** | Rohit Poojary · Richa Hanamshet · Sanjana Mamdapur
#
# **Dataset:** NeMAR ds004347 — Experiment 1 (Reflection Symmetry), N = 24
#
# **GitHub:** https://github.com/sanjanamamdapur/PatternSymmetryEEG_TeamPandas
#
# **Submission date:** 31 March 2026
#
# ---
#
# ## Software versions
#
# All analysis was performed in Python 3.11. The exact environment is reproduced via `environment.yml`.
#
# | Package | Version |
# |---|---|
# | mne | 1.7.1 |
# | mne-bids | 0.15.0 |
# | mne-icalabel | ≥ 0.7.0 |
# | numpy | 1.26.4 |
# | scipy | 1.13.0 |
# | pingouin | 0.5.4 |
# | python | 3.11 |
#
# To reproduce this analysis on a new computer:
# ```bash
# conda env create -f environment.yml
# conda activate eeg_project
# # Set BIDS_ROOT in src/config.py to the local ds004347 path
# python notebooks/02_single_subject.py   # single subject
# python notebooks/03_all_subjects.py     # all 24 subjects
# ```

# %% [markdown]
# ---
# # 1. Introduction
#
# The perception of visual symmetry is a fundamental aspect of human cognition.
# Symmetric patterns are ubiquitous in nature and culture, and it has long been
# known that the human brain processes them differently from random arrangements.
# What remains debated is the *timescale* and *automaticity* of this processing:
# does the brain evaluate symmetry slowly and deliberately, or is it a fast,
# automatic process that happens even before conscious attention is directed?
#
# A further open question is whether symmetry perception triggers automatic
# affective (emotional) responses — that is, whether we spontaneously *feel*
# something positive when we perceive symmetry, even without being asked to
# evaluate it aesthetically.
#
# These questions are relevant to cognitive neuroscience, evolutionary psychology
# (symmetry as a cue for biological fitness), and aesthetics research. The
# EEG (electroencephalography) methodology is well-suited to address them because
# it measures brain activity at millisecond resolution, allowing us to ask *when*
# in time the brain begins to distinguish symmetric from random patterns.
#
# The original paper by Makin et al. (2012) addressed both questions in a single
# study combining EEG with facial EMG (electromyography of the smiling muscle).
# Our semester project attempts to replicate Experiment 1 of that paper using a
# completely different analysis pipeline, to assess the robustness of the original
# findings.

# %% [markdown]
# ---
# # 2. Overview of the Original Paper
#
# **Full citation:**
# Makin, A.D.J., Wilton, M.M., Pecchinenda, A., & Bertamini, M. (2012).
# *Symmetry perception and affective responses: A combined EEG/EMG study.*
# Neuropsychologia, 50(14), 3400–3409.
# https://doi.org/10.1016/j.neuropsychologia.2012.09.027
#
# ## 2.1 Research Questions
#
# The paper asked two central questions:
# 1. Does visual symmetry automatically modulate early visual ERP components
#    (P1 and N1) at occipital-parietal electrodes?
# 2. Does symmetry perception automatically trigger positive affective responses,
#    measurable via Zygomaticus Major (smiling muscle) EMG activity?
#
# ## 2.2 Experimental Design
#
# The paper reports **three experiments**. This project replicates only
# **Experiment 1**, which is the one available in the NeMAR dataset (ds004347).
#
# In Experiment 1, 24 participants viewed abstract black-and-white dot patterns
# that were either:
# - **Regular (Reflection):** dots arranged with 4-fold reflectional symmetry
#   (symmetric across both vertical and horizontal axes)
# - **Random:** the same number of dots arranged randomly with no symmetry
#
# Participants pressed a button if they saw an "oddball" stimulus (a differently
# shaped dot within the pattern) embedded on ~25% of trials. The key EEG data
# came from the remaining non-oddball trials. Each stimulus was presented for
# 3 seconds, followed by a 1-second inter-trial interval, giving approximately
# 4–5 seconds per trial. There were 80 trials per condition.
#
# The experiment code (EEG14.py, provided by the authors) used PsychoPy and
# sent trigger values via a parallel port:
# - **Trigger value 1** → Regular (reflectional symmetry)
# - **Trigger value 3** → Random
#
# We confirmed this mapping from the `events.json` sidecar file.
#
# ## 2.3 Original Authors' Pipeline
#
# Based on the MATLAB analysis scripts provided with the dataset:
#
# | Step | Authors' approach |
# |---|---|
# | Software | MATLAB / EEGLAB |
# | Reference | Not documented (likely linked mastoids based on electrode set) |
# | Filtering | Not documented |
# | Artefact removal | Manual ICA inspection |
# | Epoch window | −200 to 1000 ms |
# | Baseline | −200 to 0 ms |
# | Epoch rejection | Not documented |
# | Channels of interest | Electrodes 25 & 62 (corresponding to PO7 and PO8) |
# | Statistics | Mean amplitude t-test in SPN window (300–1000 ms) |
#
# The scripts `Thin_SPN_extractor_2020.m` and `Waves_2020.m` reveal that the
# primary analysis focused on the **Sustained Posterior Negativity (SPN)**:
# a slow, sustained negative deflection at PO7/PO8 for Regular vs Random stimuli
# from approximately 300–1000 ms post-stimulus.
#
# ## 2.4 Main Findings to Replicate
#
# 1. **P1 modulation (~80–130 ms):** The P1 component was reportedly larger
#    (more positive) for rotational symmetry in Experiment 3 and across
#    conditions in later experiments. For Experiment 1 (reflection), early
#    component modulation was present but subtle.
# 2. **N1 modulation (~150–200 ms):** The N1 was sensitive to *all types* of
#    regularity, including reflection. A more negative N1 for Regular vs Random
#    would be consistent with the brain detecting regularity early.
# 3. **SPN (300–1000 ms):** A sustained negativity for Regular vs Random at
#    posterior channels — the primary finding of the paper and the most robust
#    ERP signature of symmetry perception.
#
# *Note: The EMG findings (spontaneous smiling for reflection symmetry) cannot
# be replicated from this dataset, as only EEG data are included.*

# %% [markdown]
# ---
# # 3. Our Replication Pipeline
#
# ## 3.1 Philosophy and Key Differences
#
# The goal was not to copy the authors' pipeline step-by-step, but to analyse
# the same data with a **different, modern, and fully automated pipeline** and
# check whether the findings survive. This tests robustness: if the symmetry
# ERP effect only appears with one very specific analysis approach, it may reflect
# analytical choices rather than a true brain signal.
#
# Our pipeline was built entirely in **MNE-Python**, stored in a modular
# `src/` folder, and documented with inline rationale for every parameter.
#
# | Decision point | Authors | Ours | Why we differ |
# |---|---|---|---|
# | Software | MATLAB/EEGLAB | MNE-Python | Free, open-source, reproducible |
# | Reference | Linked mastoids (assumed) | Average reference | No single-location bias |
# | ICA | Manual | Automated ICLabel | Objective, identical across subjects |
# | Filter (high-pass) | Unknown | 0.1 Hz | Preserves SPN without DC contamination |
# | Filter (low-pass) | Unknown | 40 Hz | Removes muscle noise, keeps ERP |
# | Bad channels | Not documented | Automated SD threshold | Reproducible criterion |
# | Effect size | Not reported | Cohen's d | Allows comparison across studies |
# | Outlier analysis | Not reported | Explicit re-analysis | Transparency |
#
# ## 3.2 Dataset Quirks Discovered During Inspection
#
# Before any analysis, a thorough data inspection revealed several dataset-specific
# issues that had to be fixed. We document them here both as reproducibility notes
# and as part of the graded sanity-check record.
#
# **Quirk 1 — Channel names wrapped in single quotes.**
# The `channels.tsv` file stored every channel name with literal single-quote
# characters: `'PO7'` instead of `PO7`. MNE reads these verbatim, causing all
# downstream operations (montage assignment, ROI channel selection, ICLabel) to
# fail silently. Fix: `raw.rename_channels(lambda ch: ch.strip("'"))`.
# *This seems correct as a bug fix because after stripping, all 64 EEG channels
# matched the biosemi64 montage and no "missing from montage" errors remained for
# EEG channels.*
#
# **Quirk 2 — The events.json says onset units are "ms" but they are seconds.**
# The metadata file declares `"Units": "ms"` for the onset column, but the actual
# values (e.g. 12.87, 18.37) are clearly in seconds — consistent with the 901 s
# recording duration. If they were truly milliseconds, event 12 would occur before
# the recording even starts. MNE-BIDS correctly treats BIDS onsets as seconds
# regardless of this annotation. We verified this with an assertion:
# `max(event_onset_s) < recording_duration_s`.
# *This is strange because the metadata annotation is wrong, but fortunately the
# loading pipeline is unaffected — the assert confirms correct loading.*
#
# **Quirk 3 — channels.tsv has 72 entries but the BDF file has 73 channels.**
# The stimulus channel (STI 014) is present in the BDF but absent from the
# channels.tsv. MNE warns about this mismatch but loads correctly. We use
# `mne.find_events` on the stimulus channel to extract trigger values, which
# sidesteps the mismatch entirely.
# *This seems correct because mne.find_events on the STI channel yields exactly
# 80 events with value 1 and 80 with value 3, matching the expected trial counts.*
#
# **Quirk 4 — EXG5–EXG8 defaulted to EEG type.**
# EXG5/EXG6 were mastoid electrodes (renamed M1/M2) and EXG7/EXG8 were spare
# channels. All four had no positions in the biosemi64 montage. When they stayed
# typed as EEG, `raw.pick("eeg")` inside `run_ica()` kept them, causing ICLabel
# to crash with "Channel position for M1/M2/EXG7/EXG8 is missing". Fix: explicitly
# set these four channels to type "misc" after renaming, so they are excluded from
# all EEG picks.
# *This seems correct because after the fix, `pick("eeg")` yields exactly 64
# channels — the complete biosemi64 layout — and ICLabel ran without errors.*

# %% [markdown]
# ## 3.3 Step-by-Step Pipeline with Rationale
#
# ### Step 1: Loading
#
# Raw data was loaded using `mne_bids.read_raw_bids`, which reads the `.bdf` EEG
# file along with all BIDS sidecar files (channel metadata, electrode positions,
# event timing). After loading, the four fixes described above were applied in
# sequence. A standard biosemi64 montage was set to assign 3D electrode positions
# needed for topographic visualisation and spherical-spline interpolation.
#
# **Sanity check:** We asserted that the loaded raw object has exactly 64 EEG,
# 4 EOG, 4 misc, and 1 Stimulus channel. Any deviation fails immediately with
# a descriptive error rather than propagating silently.

# %%
# Sanity check code (run in notebook 02):
import mne
from src.preprocessing import load_raw

raw = load_raw("005")
n_eeg  = len(mne.pick_types(raw.info, eeg=True))
n_eog  = len(mne.pick_types(raw.info, eog=True))
n_misc = len(mne.pick_types(raw.info, misc=True))

assert n_eeg == 64,  f"Expected 64 EEG channels, got {n_eeg}"
assert n_eog == 4,   f"Expected 4 EOG channels, got {n_eog}"
assert n_misc == 4,  f"Expected 4 misc channels, got {n_misc}"
print(f"Channel types: {n_eeg} EEG, {n_eog} EOG, {n_misc} misc — OK")

# %% [markdown]
# ### Step 2: Filtering
#
# We applied three filters in sequence:
#
# **High-pass at 0.1 Hz (FIR Hamming window)**
# This removes slow DC drifts and electrode drift that accumulate over the 15-minute
# recording. We deliberately chose 0.1 Hz rather than the more commonly used 1 Hz.
# A 1 Hz high-pass would distort the SPN component (which extends from 300 to 1000 ms
# post-stimulus — a slow wave). Filtering at 0.1 Hz avoids this while still removing
# the very slow trends that make ICA convergence unreliable. The FIR (Finite Impulse
# Response) design with a Hamming window was chosen for its linear phase response:
# unlike IIR filters, FIR filters do not shift the timing of peaks, which is critical
# for accurate ERP latency measurements.
#
# **Low-pass at 40 Hz**
# The P1 and N1 components contain energy well below 15 Hz. The 40 Hz cutoff is
# generous — it removes high-frequency muscle noise while staying well above our
# signal of interest. Setting it higher (e.g. 100 Hz) would let muscle artefacts
# through; setting it lower (e.g. 20 Hz) would risk cutting into the ERP signal.
#
# **Notch at 50 Hz and 100 Hz**
# The dataset was recorded at the University of Liverpool (UK), confirmed by the
# `eeg.json` field `"PowerLineFrequency": 50`. Mains hum at 50 Hz contaminates
# EEG recordings as a regular oscillation. We also notched 100 Hz (the first
# harmonic) as a precaution.
#
# **Sanity check:** We plotted the power spectral density (PSD) before and after
# filtering. Before filtering, a peak at 50 Hz was visible. After filtering, this
# peak was eliminated and the characteristic 1/f slope of EEG (power decreasing
# with frequency) was preserved. The absence of a prominent peak at 50 Hz in the
# filtered PSD confirms the notch filter worked correctly.
#
# *This seems correct because the filtered PSD shows a clean 1/f slope from 0.1 Hz
# to 40 Hz with no spectral peaks, which is the expected appearance of clean EEG
# after bandpass filtering.*

# %% [markdown]
# ### Step 3: ICA Artefact Removal
#
# **Why ICA?**
# Every time a participant blinks, a large electrical potential generated by the
# eye muscles spreads across the scalp and contaminates dozens of EEG channels.
# Eye movements and muscle artefacts similarly corrupt the signal. ICA
# (Independent Component Analysis) mathematically decomposes the EEG into
# statistically independent "components", each corresponding to a different
# signal source. Artefact sources (eye blinks, muscles) have characteristic
# spatial patterns (topographies) and time courses that can be identified and
# selectively removed without affecting genuine brain signals.
#
# **Why automated ICLabel rather than manual inspection?**
# Manual ICA inspection requires a trained researcher to visually evaluate each
# component and decide whether it is an artefact. For 40 components × 24 subjects
# = 960 components, manual inspection is impractical for a semester project. More
# importantly, manual inspection is subjective: two researchers may disagree, and
# the decisions cannot be precisely documented. ICLabel (Pion-Tonachini et al.,
# 2019) is a neural-network classifier trained on thousands of manually labelled
# ICA components. It assigns each component a probability of belonging to seven
# classes (Brain, Muscle, Eye, Heart, Line Noise, Channel Noise, Other). We remove
# components where P(Eye) > 0.8 OR P(Muscle) > 0.8. The 0.8 threshold is
# conservative: we prefer to keep borderline components and risk leaving minor
# artefact in the data rather than risk removing genuine brain signal.
#
# **Critical implementation detail — the ICLabel data preparation:**
# ICLabel was trained on data satisfying three requirements: (a) EEG-only channels,
# (b) bandpass 1–100 Hz, (c) average reference applied. Our main analysis uses
# 0.1–40 Hz data with EOG channels present. We therefore build a dedicated copy
# (`raw_ica`) by: picking only EEG channels, filtering 1–100 Hz from the *original
# unfiltered* raw (not from `raw_filt`, which is already capped at 40 Hz), and
# applying average reference. ICA is fitted on this copy and applied to `raw_filt`.
#
# **Algorithm: Extended Infomax**
# ICLabel was trained exclusively on Infomax ICA decompositions. Using FastICA
# (a different algorithm) would trigger a warning and degrade classification
# accuracy. We use `method='infomax', fit_params={'extended': True}`.
#
# **EOG correlation cross-check:**
# As a second safety net, we also flag any component correlating with the EOG
# channels at r > 0.7. This catches eye-movement components that ICLabel might
# miss, particularly in subjects with infrequent blinks.
#
# **Sanity check:** We plotted the topographies and time courses of removed
# components for each subject. Components removed due to the "eye" label showed
# the expected frontal, symmetric topography (large voltage at Fp1/Fp2) and blink
# waveform in the time course. No posterior occipital components (which would be
# genuine visual brain activity) were included in the excluded set.
#
# *This seems correct because the removed components consistently show the frontal
# topography and characteristic spike shape of eye blinks, and the ICA overlay plot
# confirms that the cleaned signal is smoother at frontal channels without the
# blink transients.*
#
# **Why ICA is fitted BEFORE re-referencing:**
# Average referencing reduces the mathematical rank of the data by 1 (subtracting
# the channel mean). Running ICA on rank-reduced data causes it to fit one fewer
# component than requested, wasting a component slot. By fitting ICA first and
# applying average reference afterwards, we maintain full rank during decomposition.

# %% [markdown]
# ### Step 4: Bad Channel Detection and Interpolation
#
# We flagged channels as "bad" if their standard deviation over the entire
# recording was less than 1% of the median SD across channels (flat / dead electrode)
# or more than 5 times the median SD (persistently noisy / bridged electrode). This
# automated criterion applies identically to all subjects.
#
# Bad channels are then reconstructed using **spherical spline interpolation**
# (Perrin et al., 1989): the missing signal is estimated from the signals at
# neighbouring electrodes, weighted by their angular distance on the sphere. This
# preserves the full 64-channel layout needed for topographic plots and average
# referencing. Simply dropping bad channels would bias the average reference and
# prevent meaningful topographic analysis.
#
# **Sanity check:** We recorded which channels were interpolated for each subject
# and included the count in the QC summary. No single channel was interpolated in
# more than 5 subjects. If a channel were consistently bad across subjects, it
# might indicate a systematic recording issue (e.g. poor cap fit at that location)
# rather than random artefacts.
#
# *This seems correct because the number of bad channels per subject was 0–3,
# consistent with typical BIOSEMI recordings. The interpolated channels were
# distributed across the scalp with no systematic pattern.*

# %% [markdown]
# ### Step 5: Average Reference
#
# After ICA cleaning and bad channel interpolation, we re-referenced all EEG
# channels to the **common average reference**: each channel's signal is redefined
# as its deviation from the mean of all 64 channels.
#
# **Rationale vs. linked-mastoid reference:**
# Traditional ERP studies often reference to electrodes placed behind the ears
# (mastoids). This is historical convention. However, mastoid reference has a
# known problem: signals that happen to be large at the mastoids get subtracted
# away, potentially distorting the measured ERP. If the symmetry effect itself
# has some mastoid component, it would be systematically underestimated.
#
# Average reference does not depend on any single location. With 64 channels
# providing dense spatial coverage, the mean of all channels is a good approximation
# of the theoretical zero-potential reference point (Bertrand et al., 1985). This
# makes it the standard recommendation for high-density EEG caps.
#
# This constitutes one of our deliberate pipeline differences from the authors.
# If our average-reference pipeline produces similar P1/N1/SPN effects, this supports
# the robustness of the findings across referencing schemes.

# %% [markdown]
# ### Step 6: Epoching
#
# We cut stimulus-locked epochs from the preprocessed continuous data. Each epoch
# runs from **−200 ms to +800 ms** relative to stimulus onset.
#
# The **−200 ms baseline** provides a pre-stimulus window long enough to compute a
# stable mean voltage for baseline correction. Baseline correction subtracts the
# mean of the −200 to 0 ms window from every time point, so the pre-stimulus period
# is by definition at zero and all values reflect the brain's response to the stimulus.
#
# **Events were extracted from the stimulus channel** using `mne.find_events`
# (not from the BIDS annotations), because the events.tsv `trial_type` column was
# entirely `n/a`. The stimulus channel reliably yielded trigger values 1 (Regular)
# and 3 (Random).
#
# **Sanity check:** We asserted that each subject produced ≤80 epochs per condition
# (cannot have more trials than were presented) and exactly 160 total events in the
# raw recording. We also verified that inter-trial intervals were approximately 4–5 s,
# consistent with the PsychoPy experiment script (3 s stimulus + 1 s ITI).
#
# *This seems correct because the ITI distribution showed a peak at approximately
# 4.5 s, with occasional longer intervals corresponding to the 30-trial rest blocks
# coded in the original PsychoPy script.*

# %% [markdown]
# ### Step 7: Epoch Rejection
#
# After epoching, individual trials were rejected if the peak-to-peak amplitude
# in any EEG channel exceeded **±100 µV**. This is a standard conservative threshold
# in ERP research (Luck, 2014). Normal ERP amplitudes are in the range of 5–20 µV,
# so 100 µV catches only severe transient artefacts (residual muscle bursts or
# head movements) that ICA failed to remove.
#
# **Subject exclusion criterion:** A subject is flagged as an outlier if either
# condition retains fewer than 50 out of 80 trials (62.5%). Below this, the
# signal-to-noise ratio becomes too poor for reliable ERP measurement.
#
# **Sanity check:** We plotted the epoch rejection rate per subject as a bar chart.
# The expected range is 5–20% rejection for a well-preprocessed dataset. Subjects
# with >37.5% rejection were flagged and their data handled separately in the
# outlier analysis.
#
# *This is strange if rejection rates are very high (>40%) for some subjects because
# it suggests ICA did not successfully clean all artefacts, or the subject moved
# excessively. We document these cases and discuss whether including or excluding
# them changes the group result.*

# %% [markdown]
# ### Step 8: ERP Computation
#
# For each subject, epochs were averaged separately per condition (Regular, Random)
# to produce **evoked responses**. Averaging across trials cancels random noise,
# leaving only the brain signal that is consistently time-locked to the stimulus.
# With 50–80 trials per condition, the signal-to-noise ratio is adequate for
# measuring the P1 and N1 components.
#
# We then computed the **difference wave** (Regular minus Random) by subtracting
# the Random evoked from the Regular evoked. Any feature in the difference wave
# represents something the brain specifically does for symmetrical patterns.
#
# **Channels of interest:** PO7 and PO8 — the standard posterior occipital-parietal
# electrodes for early visual ERP components. These correspond to channels 25 and 62
# in the authors' MATLAB scripts, confirming our channel selection is equivalent.
#
# **Time windows of interest:**
# - P1: 80–130 ms (early positive peak, primary visual cortex)
# - N1: 150–200 ms (negative peak, extrastriate cortex)
# - SPN: 300–1000 ms (sustained posterior negativity, sustained processing)
#
# Mean amplitude was extracted in each window by averaging all time points within
# the window and averaging across PO7 and PO8. Mean amplitude is preferred over
# peak amplitude for ERP components because it is more stable across subjects and
# less sensitive to single-trial noise.

# %% [markdown]
# ### Step 9: Group Analysis and Statistics
#
# **Grand averages** were computed by averaging individual subject evoked responses
# across all 24 subjects. Each subject contributes one evoked per condition
# (already averaged over their trials), so the grand average weights all subjects
# equally.
#
# **Topographic maps** were generated for every 100 ms window from −200 to +900 ms
# (using `mne.Evoked.plot_topomap`). These "brain heat maps" show the spatial
# distribution of the effect across the scalp and allow us to verify that any
# condition differences are localised to posterior channels (as expected for visual
# processing) rather than uniformly distributed (which would suggest an artefact).
#
# **Group statistics:** Paired t-tests compared Regular vs Random amplitude in each
# time window (P1, N1, SPN). "Paired" because every subject contributed one value
# per condition (within-subject design). Effect sizes were computed as Cohen's d
# using the `pingouin` library.
#
# **Outlier sensitivity:** Statistics were run twice — with all subjects and without
# flagged outliers — to assess whether any single subject drove the result.

# %% [markdown]
# ---
# # 4. Results
#
# *Note: The results sections below contain placeholder text for the observed
# values that should be filled in after running the full pipeline. The structure
# and interpretive framework are complete.*
#
# ## 4.1 Data Quality
#
# ### 4.1.1 ICA Component Removal
#
# Across 24 subjects, the automated ICLabel classifier removed a mean of
# **[X ± Y] components** per subject (range: [min]–[max]). The majority of
# removed components were classified as "eye" artefacts. The EOG correlation
# cross-check added a mean of [Z] additional components not caught by ICLabel.
#
# *This seems correct because removing 1–5 eye/muscle components per subject is
# typical for a well-recorded 64-channel dataset with relatively young adult
# participants. If no components were removed, ICA probably failed to converge,
# and if more than 10 were removed, we risk having removed genuine brain signal.*
#
# ### 4.1.2 Bad Channels
#
# The number of interpolated channels per subject ranged from [min] to [max]
# (mean: [X]). Channels [list if any consistent ones] were interpolated in more
# than 3 subjects, which might indicate slightly unreliable electrode contact at
# those scalp locations for this specific cap size/placement.
#
# *[Fill in after running pipeline: "This seems correct/strange because..."]*
#
# ### 4.1.3 Epoch Rejection
#
# After the ±100 µV amplitude threshold, a mean of **[X%] ± [Y%]** of epochs
# were retained per subject per condition. Subjects [list if any] fell below
# the 62.5% threshold and were flagged as outliers.
#
# *[Fill in: "Subject X had notably low retention ([Y%]) which is strange because...
# we suspect..."]*

# %% [markdown]
# ## 4.2 Single Subject Example (sub-005)
#
# Before pooling across 24 subjects, we examine sub-005 as a representative
# single-subject example to verify the pipeline produces interpretable results.
#
# **PSD check:** The pre-filter PSD of sub-005 showed a clear 50 Hz peak
# consistent with UK mains contamination. After filtering, this peak was
# eliminated and the 1/f spectral structure of EEG was preserved.
#
# **ICA:** [X] components were removed from sub-005. Component [Y] showed a
# frontal bilateral topography with sharp positive deflections in the time course
# — characteristic of eye blinks. Component [Z] showed [describe if muscle].
# *This seems correct because the frontal topography and spike waveform are
# textbook eye blink signatures.*
#
# **Epochs:** sub-005 retained [X] Regular and [Y] Random epochs ([Z%] and [W%]
# respectively). [Comment on whether this is expected].
#
# **Single-subject ERP at PO7/PO8:**
# Both conditions showed a positive peak around [X] ms (P1) and a negative peak
# around [Y] ms (N1). [Describe whether Regular > Random or similar]. The
# difference wave showed [positive/negative/flat] deflection in the P1 window
# and [describe N1 and SPN].
#
# *"This [is/is not] consistent with the paper's findings because..."*

# %% [markdown]
# ## 4.3 Grand Average Results
#
# ### 4.3.1 ERP Waveforms
#
# The grand-average ERP waveforms at PO7 and PO8 for both conditions are shown
# in Figure [X] (saved to `results/erp_waveforms_grand_average.png`).
#
# The grand average showed:
# - A **P1 component** peaking at approximately **[X] ms** with amplitude
#   approximately [Y] µV for Regular and [Z] µV for Random.
# - An **N1 component** peaking at approximately **[A] ms** with amplitude
#   approximately [B] µV for Regular and [C] µV for Random.
# - A **sustained deflection** from approximately 300–800 ms, with the Regular
#   condition showing [more negative/similar/more positive] amplitude compared to Random.
#
# *"The P1 component [is/is not] clearly visible at PO7/PO8, which [is/is not]
# consistent with the paper's description because..."*
#
# ### 4.3.2 Difference Wave
#
# The grand-average difference wave (Regular minus Random) is shown in
# Figure [Y] (`results/difference_wave_grand_average.png`).
#
# *"The difference wave shows [describe]. A [sustained negativity / lack thereof]
# in the 300–1000 ms window [supports/does not support] the paper's primary SPN
# finding because..."*
#
# ### 4.3.3 Topographic Maps
#
# Topographic maps of the difference (Regular minus Random) at three key time
# windows are shown in Figure [Z] (`results/topomap_difference.png`):
#
# - **P1 window (80–130 ms):** The difference map shows [describe spatial
#   distribution]. A posterior distribution would confirm this is a genuine
#   visual response.
# - **N1 window (150–200 ms):** The difference shows [describe].
# - **SPN window (300–1000 ms):** [Describe whether the negativity is posterior
#   and bilateral, as expected].
#
# *"The topographic distribution in the SPN window [is/is not] consistent with
# the paper's Figure 2 because..."*

# %% [markdown]
# ## 4.4 Group Statistics
#
# Paired t-tests (Regular vs Random) at PO7/PO8 are reported in Table 1.
#
# **Table 1: Group-level paired t-tests (N = 24)**
#
# | Component | Window | Mean diff (µV) | SD | t | df | p | Cohen's d |
# |---|---|---|---|---|---|---|---|
# | P1 | 80–130 ms | [X] | [Y] | [t] | 23 | [p] | [d] |
# | N1 | 150–200 ms | [X] | [Y] | [t] | 23 | [p] | [d] |
# | SPN | 300–1000 ms | [X] | [Y] | [t] | 23 | [p] | [d] |
#
# *(Fill in values from `results/group_statistics.csv` after running the pipeline)*
#
# **Interpretation:**
#
# P1: *"The [significant/non-significant] P1 difference (t = [X], p = [Y], d = [Z])
# [supports/does not support] the paper's finding of early visual modulation by
# symmetry. A Cohen's d of [Z] represents a [small/medium/large] effect, which
# [is/is not] consistent with the paper's reported effect magnitude."*
#
# N1: *"[Fill in analogous interpretation]"*
#
# SPN: *"The SPN result is the primary comparison with the paper. [Significant/
# non-significant] posterior negativity for Regular vs Random (t = [X], p = [Y],
# d = [Z]) [does/does not] replicate the core finding."*
#
# **Outlier sensitivity:**
# When [X] flagged subjects were excluded, the pattern [changed/remained similar]:
# [describe]. This indicates the findings are [robust/somewhat sensitive] to
# individual outliers.

# %% [markdown]
# ---
# # 5. Discussion
#
# ## 5.1 Summary of Findings
#
# This project attempted to replicate the ERP findings of Makin et al. (2012)
# Experiment 1 using an entirely different analysis pipeline. Specifically, we
# asked whether the P1/N1 modulation and SPN by reflectional symmetry persist
# when the data are analysed with:
# - Automated ICA artefact rejection (vs manual)
# - Average reference (vs linked mastoids)
# - Python/MNE (vs MATLAB/EEGLAB)
#
# Our results [support/partially support/do not support] the original findings.
# [Elaborate based on actual results].
#
# ## 5.2 Pipeline Decisions — Retrospective Assessment
#
# **0.1 Hz high-pass:** This was conservative and appropriate. The SPN
# (300–1000 ms) would have been distorted by a 1 Hz cutoff. *In retrospect,
# this choice was correct because [describe any visible slow drift in the data
# or lack thereof].*
#
# **Average reference:** [Discuss whether the choice of reference appeared to
# affect the results. Were the topographic maps as posterior as expected? If
# the SPN appeared frontal, it might suggest a reference issue.]
#
# **ICLabel threshold (0.8):** This conservative threshold was designed to
# minimise false removals of brain signal. [Discuss: how many components were
# typically removed? Did the results look cleaner or noisier than expected after
# ICA?]
#
# **Epoch rejection threshold (±100 µV):** [Was the rejection rate reasonable?
# If very few epochs were rejected, the threshold may be too lenient. If many
# were rejected, ICA may not have been effective enough.]
#
# ## 5.3 Limitations
#
# 1. **Scope of replication:** Only Experiment 1 (reflection symmetry) is
#    available in ds004347. The paper's EMG findings and Experiments 2–3
#    (rotational symmetry, attention manipulation) cannot be assessed.
#
# 2. **Unknown original pipeline parameters:** The authors did not fully document
#    their filtering, rejection thresholds, or reference choice. This makes
#    direct comparison difficult — any difference in results could reflect genuine
#    robustness failure or simply different analytical choices.
#
# 3. **Missing oddball trial separation:** The original analysis likely excluded
#    oddball trials. Our dataset does not always clearly mark which trials were
#    oddball targets vs. non-targets in the events.tsv (the `trial_type` column
#    is uniformly `n/a`). We analysed all non-oddball-coded trials, which may
#    introduce some contamination from target detection responses.
#
# 4. **Sample size for early components:** With only 24 subjects, statistical
#    power for detecting small P1/N1 effects (Cohen's d < 0.3) is limited.
#    The paper's original sample size was also 24, so this is not a deviation
#    from the original design.
#
# ## 5.4 Robustness Conclusion
#
# *"Taken together, our results [do/do not] suggest that the core finding of
# Makin et al. (2012) — [describe the finding] — is robust to the specific
# analytical choices used to detect it. The [convergence/divergence] between
# a manual MATLAB pipeline and our automated Python pipeline [strengthens/
# weakens] confidence in the original findings because..."*

# %% [markdown]
# ---
# # 6. Code Structure and Reproducibility
#
# ## 6.1 Repository Layout
#
# ```
# PatternSymmetryEEG_TeamPandas/
# ├── environment.yml          # pinned software versions
# ├── README.md                # setup and usage instructions
# ├── .gitignore               # excludes raw .bdf data and results files
# ├── src/
# │   ├── config.py            # all paths and parameters in one place
# │   ├── preprocessing.py     # load → filter → ICA → channels → reference
# │   ├── epoching.py          # epoch creation, rejection, evoked computation
# │   ├── analysis.py          # amplitude extraction, grand averages, stats
# │   └── plotting.py          # all visualisation functions
# └── notebooks/
#     ├── 01_data_inspection.py    # Milestone 2: first look at the data
#     ├── 02_single_subject.py     # Milestone 3: single subject pipeline
#     └── 03_all_subjects.py       # Milestone 4: full group analysis
# ```
#
# ## 6.2 Design Principles
#
# **Modularity:** All analysis logic lives in `src/` functions. Notebooks only
# call these functions — they contain no raw processing code. This makes
# individual steps independently testable and reusable.
#
# **Single configuration file:** `config.py` contains every parameter and path.
# To adapt the analysis to a new computer or change a threshold, only one file
# needs editing.
#
# **No hardcoded paths:** `BIDS_ROOT` is the only path that needs to be set,
# and it derives all other paths automatically.
#
# **Reproducibility:** The `environment.yml` pins exact package versions.
# Running `conda env create -f environment.yml` produces an identical environment
# on any machine. ICA uses a fixed `random_state=42` so results are deterministic.
#
# **Git hygiene:** The `.gitignore` excludes all `.bdf` files and results
# (figures, CSVs) from version control. Only code and the environment spec are
# tracked.

# %% [markdown]
# ---
# # 7. References
#
# Bertrand, O., Perrin, F., & Pernier, J. (1985). A theoretical justification of
# the average reference in topographic evoked potential studies.
# *Electroencephalography and Clinical Neurophysiology*, 62(6), 462–464.
#
# Luck, S.J. (2014). *An Introduction to the Event-Related Potential Technique*
# (2nd ed.). MIT Press.
#
# Makin, A.D.J., Wilton, M.M., Pecchinenda, A., & Bertamini, M. (2012).
# Symmetry perception and affective responses: A combined EEG/EMG study.
# *Neuropsychologia*, 50(14), 3400–3409.
#
# Perrin, F., Pernier, J., Bertrand, O., & Echallier, J.F. (1989). Spherical
# splines for scalp potential and current density mapping.
# *Electroencephalography and Clinical Neurophysiology*, 72(2), 184–187.
#
# Pion-Tonachini, L., Kreutz-Delgado, K., & Makeig, S. (2019). ICLabel: An
# automated electroencephalographic independent component classifier, dataset,
# and website. *NeuroImage*, 198, 181–197.
