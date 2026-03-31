# EEG Semester Project — Symmetry Perception & Affective Responses
**Team Python** | Rohit Poojary · Richa Hanamshet · Sanjana Mamdapur

## Paper
Makin, A.D.J., Wilton, M.M., Pecchinenda, A., & Bertamini, M. (2012).
*Symmetry perception and affective responses: A combined EEG/EMG study.*
Neuropsychologia. https://doi.org/10.1016/j.neuropsychologia.2012.09.027

## Dataset
NeMAR ds004347 · 24 subjects · BIOSEMI ActiveTwo · 512 Hz · 64 EEG + 8 EOG channels
https://nemar.org/dataexplorer/detail?dataset_id=ds004347

## Goal
Reproduce the P1/N1 ERP modulation by symmetry (Regular vs Random patterns)
using a pipeline that deliberately differs from the authors' to test robustness.

## Setup

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd EEGProject

# 2. Create the conda environment (takes ~3 min)
conda env create -f environment.yml
conda activate eeg_project

# 3. Set your local data path
# Open src/config.py and set BIDS_ROOT to where ds004347 lives on your machine

# 4. Run notebooks in order
jupyter lab
# Open notebooks/01_data_inspection.py  (run as notebook via jupytext)
# Open notebooks/02_single_subject.py
# Open notebooks/03_all_subjects.py
```

## Notebook format
Notebooks are stored as `.py` files with `# %%` cell markers (jupytext percent format).
This keeps them diff-friendly in Git. JupyterLab opens them as notebooks automatically
when jupytext is installed.

## Project structure
```
EEGProject/
├── environment.yml
├── README.md
├── src/
│   ├── config.py          ← set BIDS_ROOT here
│   ├── preprocessing.py   ← filtering, ICA, bad channel removal
│   ├── epoching.py        ← epoch creation, rejection, evokeds
│   ├── analysis.py        ← ERP metrics, stats
│   └── plotting.py        ← all visualisation functions
├── notebooks/
│   ├── 01_data_inspection.py
│   ├── 02_single_subject.py
│   └── 03_all_subjects.py
└── results/               ← saved figures and CSVs (git-ignored for size)
```

## Software versions (for reproducibility)
See environment.yml for exact pinned versions.
Run `mne.sys_info()` inside any notebook to confirm the active environment.
