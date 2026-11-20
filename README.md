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

Eight features per driver per race, built in `notebooks/03_features.ipynb` and saved to `races_clean.csv`. Each one for a race uses only what was known before that race, and every season-based one resets in round 1.

| Feature | Meaning | Blank or missing |
| --- | --- | --- |
| `Grid` | Starting position | Pit-lane starts go to the back |
| `PercentageDifference` | Gap from the driver's best qualifying lap to pole, in % | No time set: slowest time in that session |
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

## Model

Logistic regression on the eight features, in `notebooks/04_model.ipynb`.

### The equation

For driver-race $i$ with scaled features $x_{i1}, \dots, x_{i8}$:

$$
z_i = b + w_1 x_{i1} + w_2 x_{i2} + \dots + w_8 x_{i8} = b + \mathbf{w} \cdot \mathbf{x}_i
$$

$$
P(\text{podium}_i = 1) = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}
$$

- $\mathbf{w}$ is the 8 learned weights (`model.coef_[0]`) and $b$ is the intercept (`model.intercept_[0]`).
- $z_i$ is the log-odds: $z_i = \ln\dfrac{p_i}{1 - p_i}$. A positive weight means that feature raises the chance of a podium, and a negative weight lowers it.
- The sigmoid $\sigma$ squashes any $z$ into a probability between 0 and 1: $z = 0$ gives 0.5.
- For all rows at once, in numpy: `z = X @ w + b`, then `p = 1 / (1 + np.exp(-z))`. This matches `model.predict_proba(X)[:, 1]` (checked with `np.allclose`).

Training picks $\mathbf{w}$ and $b$ to minimise the log loss over the training rows, plus the L2 penalty (default `C=1.0`):

$$
\text{loss} = -\sum_i \Big[ y_i \ln p_i + (1 - y_i) \ln(1 - p_i) \Big] + \frac{1}{2C} \lVert \mathbf{w} \rVert^2
$$

where $y_i$ is the real `Podium` value (0 or 1). The penalty term is why the features are scaled first (see below).

### Why the features are scaled

The features have very different ranges: `ConstructorPoints` runs from 0 to about 800, `Grid` from 1 to 20, `PercentageDifference` from 0 to about 5, and `DriverShare` from 0 to 1. Each one is rescaled to mean 0 and standard deviation 1, with `(X - X.mean()) / X.std(ddof=0)`. This matches `StandardScaler` (checked with `np.allclose`).

1. **The penalty treats every coefficient the same.** `LogisticRegression` uses regularisation by default (`C=1.0`), which penalises large coefficients. How large a coefficient is depends on the feature's units. 0.004 × 400 points and 3.2 × 0.5 share move the score by the same 1.6, yet the penalty barely touches the first and hits the second hard. Unscaled, small-range features like `DriverShare` would be pushed down unfairly.
2. **Training converges faster.** When one feature is in the hundreds and another in fractions, the solver zig-zags towards the answer and can stop with a "failed to converge" warning.
3. **The coefficients can be compared.** After scaling, each coefficient is the change in score for a one standard deviation move in that feature, so a bigger `|coef|` means a feature that matters more.

Notes:

- With no penalty (`penalty=None`), scaling wouldn't change the predictions; the coefficients would just rescale to match. It is the penalty that makes scaling necessary.
- Tree models such as gradient boosting split on thresholds like `Grid <= 3`, which don't move when the units change, so they don't need scaling.
- The scaler is fitted on 2022–2024 only and then applied to 2025 and 2026, so "one standard deviation" is defined by data the model could have seen before the test seasons.
- pandas `.std()` divides by n − 1 (`ddof=1`), while numpy and `StandardScaler` divide by n (`ddof=0`). On about 2,000 rows that is a 0.05% difference, which is enough to make `np.allclose` return `False`.

## Results

### Logistic regression

Trained on 2022–2024 with default settings (`LogisticRegression()`, `C=1.0`). For each race, drivers are ranked by their podium probability: rank 1 is the predicted winner and ranks 1–3 are the predicted podium.

| Season | Races | Model: winner right | Model: podium hits (avg of 3) | Grid: pole won | Grid: top 3 on podium (avg of 3) |
| --- | --- | --- | --- | --- | --- |
| 2025 | 24 | 54.2% | 2.29 | 66.7% | 2.25 |
| 2026 so far | 20 | 60.0% | 1.90 | 70.0% | 1.80 |

The model is slightly ahead of the grid on podium hits in both seasons (+0.04 and +0.10 per race) but clearly behind it on the winner. That lead is small but goes the same way in both seasons, and a small lead is believable; a large one would point to leakage.

Each race's probabilities are rescaled to add up to 3, since a race has three podium places. This doesn't change the ranking.

Rating on 2026: a lead counts only if it is bigger than one race's worth (1/20 = 0.05), because anything smaller is a single race going the other way. By that rule the model is **behind the grid on the winner (−0.10)** and **ahead on the podium (+0.10 per race, about 2 extra podiums over 20 races)**.

Coefficients, on scaled features (a negative sign lowers the podium chance):

| Feature | Coefficient |
| --- | --- |
| `Grid` | −2.02 |
| `TeamForm` | −0.85 |
| `DriverForm` | −0.15 |
| `ConstructorPoints` | +0.12 |
| `DriverShare` | +0.12 |
| `GapToTeammate` | −0.09 |
| `Reliability` | +0.08 |
| `PercentageDifference` | +0.08 |

Grid position dominates, then team form. `PercentageDifference` has the wrong sign (a bigger gap to pole should lower the chance); it overlaps heavily with `Grid`, which takes the real effect, so its small coefficient shouldn't be read as meaningful.

What the 2025 misses show:

- **It leans towards McLaren.** It picked Norris or Piastri as winner in 19 of 24 races, because strong team form outweighs pole. That is where it loses winner calls to the grid (Japan, Imola, Italy, Qatar).
- **It predicts the podium, not the win.** A strong car starting second is a safe top 3 but usually doesn't beat the pole sitter. A separate model with `Position == 1` as the target would fit the winner question better.
- **Round 1 is squashed.** With every driver's form filled with the same training median, the probabilities in round 1 add up to about 1.3 instead of 3 in every season, and the ranking falls back to grid order.
- **Some misses no pre-race model could see:** Norris disqualified in Las Vegas and retiring in Zandvoort, Leclerc retiring in São Paulo, and Hülkenberg's podium from 19th on the grid at Silverstone.

### Other models tried

All use the same 8 features and the same split (train 2022–2024).

| Model | 2025 winner | 2025 podium | 2026 winner | 2026 podium |
| --- | --- | --- | --- | --- |
| Grid baseline | **66.7%** | 2.25 | **70.0%** | 1.80 |
| **Logistic regression (kept)** | 54.2% | **2.29** | 60.0% | 1.90 |
| Logistic regression + separate win model | 58.3% | 2.29 | 60.0% | 1.90 |
| Logistic regression + `RacesSoFar` flag | 41.7% | 2.29 | 50.0% | 1.95 |
| Gradient boosting, defaults | 29.2% | 1.96 | 35.0% | 1.90 |
| Gradient boosting, shallow (`max_depth=3`, `learning_rate=0.05`) | 37.5% | 1.92 | 40.0% | 1.95 |

- **Gradient boosting lost on both measures.** With about 1,360 training rows and only about 200 podiums, the trees learn the quirks of 2022–2024 instead of general patterns. Logistic regression's simple linear shape protects it on small, noisy data.
- **A separate win model** (`Position == 1` as the target) gained one race in 2025 and was level in 2026, which is too little evidence to switch to.
- **The `RacesSoFar` flag** was meant to fix the squashed round 1. It got the round-1 probability totals right, but lost several winner calls.

### Features tried and dropped

Each candidate was added to the 8 features on its own. Log loss scores every driver's probability in every race, so it is less noisy than counting hits (lower is better).

| Feature set | 2025 winner | 2025 podium | 2025 log loss | 2026 winner | 2026 podium | 2026 log loss |
| --- | --- | --- | --- | --- | --- | --- |
| **Current 8** | 54.2% | 2.29 | 0.211 | 60.0% | 1.90 | 0.233 |
| + `RaceGapForm` | 50.0% | 2.25 | 0.211 | 60.0% | 2.00 | 0.228 |
| + `RaceGapForm`, 1.5× lapped rule | 50.0% | 2.25 | 0.213 | 55.0% | 2.00 | 0.228 |
| + `TeamRaceGapForm` | 45.8% | 2.13 | 0.213 | 60.0% | 1.95 | 0.231 |
| + `GainedForm` | 54.2% | 2.29 | 0.210 | 60.0% | 1.90 | 0.233 |
| + `FPGap` | 50.0% | 2.29 | 0.210 | 60.0% | 1.95 | 0.232 |

What each one is (all of them use only information from before the race):

- `RaceGapForm`: % gap to the winner at the finish, averaged over the driver's previous 5 races this season. `Time` in the results is the gap to the winner for everyone except the winner, so the gap is `Time / winner's total time`. A lapped car's real gap is its `Time` plus laps down × the winner's average lap. Retirements are blank, and gaps are clipped at 5% so one safety-car race or one disaster can't dominate.
- `RaceGapForm`, 1.5× lapped rule: the same, but a lapped car's gap is laps down × 1.5 × its team's best qualifying lap.
- `TeamRaceGapForm`: `RaceGapForm` averaged over both of the team's cars.
- `GainedForm`: average of grid minus finish over the previous 5 races (positions gained in the race).
- `FPGap`: gap from the driver's best lap to the fastest lap in FP2 that weekend, in % (FP1 on sprint weekends, which have only one practice session). Already in `practice.csv`, so it needed no new download.

Why none of them was kept:

- **The differences are smaller than one race.** One race moves the winner rate by 4.2 points in 2025 (1/24) and 5 points in 2026 (1/20). One podium moves the average by 0.04 or 0.05. Every change in the table is one or two races either way, and the log losses differ only in the third decimal place.
- **`RaceGapForm` is `DriverForm` again in different units.** It correlates 0.75 with `DriverForm` and 0.74 with `TeamForm`. When it is added, `DriverForm`'s coefficient moves from −0.15 to +0.05: the model shifts the same signal from one column to the other, and no new information comes in.
- **How lapped cars are handled doesn't matter.** The 1.5× rule and the `Time` + laps-down version correlate 0.93 and score almost the same. The back of the grid almost never reaches the podium, so the problem is not the cleaning but the overlap with form.
- **`GainedForm` has no signal.** Its coefficient is −0.03 and it is uncorrelated with everything else.
- **`FPGap` has the right sign (−0.18)** but doesn't improve the scores. A single best lap in practice is mostly a qualifying simulation, which the grid already captures.

The lesson: race results and qualifying data have run out of new information. Anything built from them overlaps with grid and form. Beating the grid would need information it doesn't already have, such as race pace from long runs in FP2 (lap-by-lap data, not downloaded yet).

### Frozen model

Logistic regression on the 8 features, `LogisticRegression()` with default settings (`C=1.0`, L2 penalty), features scaled with `StandardScaler`, each race's probabilities rescaled to add up to 3. No tuning after this point.

For the Las Vegas prediction it is refitted on 2022 up to the latest 2026 race, and the features for the Vegas drivers are built with the same functions as the training data, from qualifying and earlier races only.

Still to come:

- Dry run on 2026 São Paulo (round 20), treated as if the race hadn't happened yet. The features built from qualifying must match the ones in `races_clean.csv`.
- The Las Vegas prediction, committed before the race.
- After the race: a conditional logit (a softmax over the drivers in each race), which compares drivers against each other instead of scoring each one alone.