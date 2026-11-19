# F1 Podium Predictor

A model that predicts each driver's chance of finishing on the podium, using qualifying and recent-form data. Built to make a prediction for the 2026 Las Vegas Grand Prix before the race. The day-by-day plan is in [ROADMAP.md](ROADMAP.md).

## Data

Every session from 2022 up to the latest 2026 race, downloaded with [FastF1](https://docs.fastf1.dev/).

- `download.py` saves one CSV per session to `data/raw/` and skips sessions already saved, so it can be stopped and rerun. Practice and sprint qualifying need an F1 TV login; races, qualifying and sprints don't.
- `notebooks/02_build_dataset.ipynb` combines them into `data/results/`: `results.csv` (races), `quali.csv`, `sprint.csv`, `sprint_quali.csv` and `practice.csv`.

Columns added so far:

| Column | Table | Meaning |
| --- | --- | --- |
| `Best` | quali | Fastest of a driver's Q1, Q2 and Q3 laps, in seconds |
| `Podium` | results | 1 if the driver finished in the top 3 (the target) |
| `ClassifiedFlag` | results | 0 if retired, withdrawn, disqualified, excluded or not classified |
| `Grid` | results | Starting position, with pit-lane starts moved to the back of that race's grid |

Data fixes, applied in the notebook so `data/raw/` stays as downloaded:

- 2025 Miami (round 6): FastF1 has no Q1–Q3 times for Verstappen, who took pole. Entered by hand from the official results on formula1.com (1:26.870, 1:26.643, 1:26.204).

Known issue: `Best` mixes Q1, Q2 and Q3 laps, so in wet or changing conditions the gap to pole for drivers knocked out early is inflated (up to 45%). Left uncapped for now; to revisit if it hurts the model.

Train on 2022–2024, test on 2025, check on 2026 so far. The data is never shuffled.

## Baseline

The model has to beat simply trusting the starting grid:

| Season | Races | Pole sitter won | Grid top 3 on the podium (avg of 3) |
| --- | --- | --- | --- |
| 2025 | 24 | 66.7% | 2.25 |
| 2026 so far | 20 | 70.0% | 1.80 |

In 2026, pole still converts to a win about as often as in 2025, but fewer of the top three starters reach the podium.

## Features

Eight features per driver per race, all in `notebooks/02_build_dataset.ipynb`. Each one for a race uses only what was known before that race, and every season-based one resets in round 1.

| Feature | Meaning | Blank or missing |
| --- | --- | --- |
| `Grid` | Starting position | Pit-lane starts go to the back |
| `Percentage Difference` | Gap from the driver's best qualifying lap to pole, in % | No time set: slowest time in that session |
| `GapToTeammate` | Gap to the teammate's best qualifying lap, in % (negative = faster) | No teammate (2024 Australia, Albon): 0 |
| `DriverForm` | Average finish over the driver's previous 5 races this season | Round 1: median of 2022–2024 |
| `TeamForm` | Same, averaged over both of the team's cars | Round 1: median of 2022–2024 |
| `ConstructorPoints` | Team's points this season going into the race, counting that weekend's sprint | Round 1: 0 |
| `DriverShare` | Driver's share of those team points | Team on 0 points: 0.5 |
| `Reliability` | Share of the driver's earlier races this season that they finished classified | Round 1: median of 2022–2024 |

Medians are taken from the training years only, so the 2025 and 2026 test data doesn't leak into the fill values. Qualifying features are merged onto race rows; a driver who qualified but didn't start drops out.

Leakage checks:

- By hand: Norris's `DriverForm` going into 2026 São Paulo is (19 + 9 + 4 + 5 + 5) / 5 = 8.4, and the McLaren `GapToTeammate` there is −3.208%. Both match the code.
- Points: McLaren go into the 2025 finale on 800 points and score 33 there, matching the official 833.
- Scramble test: changing one race's results changed no feature for that race or any earlier race.

## Results

To be added: model scores against the baseline on 2025 and 2026, then the Las Vegas prediction.