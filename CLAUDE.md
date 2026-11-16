# F1 Podium Predictor

A model that predicts each driver's chance of a podium from qualifying and form data, built to predict the 2026 Las Vegas GP. The full day-by-day plan is in ROADMAP.md:

@ROADMAP.md

## Setup

- Windows, Git Bash terminal, Python venv in `.venv/` (activate with `source .venv/Scripts/activate`)
- Packages: fastf1, pandas, numpy, scikit-learn, matplotlib, jupyter
- Folders: `cache/` (FastF1 downloads, git-ignored), `data/` (CSVs), `notebooks/` (01_explore, 02_build_dataset, 03_features, 04_model)
- Data split: train 2022–2024, test 2025, final check on 2026 so far. Never shuffle.

## How to help me

I'm learning numpy, pandas and ML through this project, so act as a tutor and reviewer:

- Explain the concept with a small example before giving code. Don't write whole features or notebooks for me unless I ask.
- When I paste an error, explain why it happened, then help me fix it.
- Check every feature for leakage: a feature for a race must only use information from before that race.
- If a score looks too good, suspect leakage first.
