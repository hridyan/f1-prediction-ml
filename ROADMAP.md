# F1 Podium Predictor: Las Vegas GP Sprint

Nov 16, 2026 · Hridyan

## Goal and deadline

By Sunday 22 Nov, 12:00 Singapore time, a podium prediction for the Las Vegas GP is committed to GitHub, made by a model you built and can explain, with numpy learned along the way.

The real deadline is qualifying, not the race: grid position and qualifying gap are your strongest features, so the model can only run once qualifying is done. That leaves five working days.

| Session | Las Vegas local | Singapore time |
| --- | --- | --- |
| Qualifying | Fri 20 Nov, 20:00 | Sat 21 Nov, 12:00 |
| Race | Sat 21 Nov, 20:00 | Sun 22 Nov, 12:00 |

Times from [pitwall.app](https://pitwall.app/races/2026-las-vegas-grand-prix); recheck on the official F1 site on Friday in case a session moves.

Plan on 2 to 2.5 hours an evening, Monday to Friday, plus Saturday afternoon. Saturday is where the prediction happens, so everything that can be built without the Vegas grid has to be finished on Friday.

## What changes from the one-week roadmap

The one-week plan is squeezed into five days, and each day gains a short numpy lesson done on your own F1 data rather than toy examples.

| Item | One-week roadmap | This sprint |
| --- | --- | --- |
| Length | 7 days, about 2 hours each | 5 evenings plus Saturday afternoon |
| numpy | "You will barely touch it" | 20 to 30 minutes a day, then used in the real code |
| Features | 12 in three batches | 8: grid, gap to pole, gap to teammate, driver form, team form, constructor points, driver share of team points, reliability |
| Track features | Three, on day 6 | One optional: Vegas history. Only 2023 to 2025 exist, so it is weak evidence |
| Models | Logistic regression, then gradient boosting | Same, but gradient boosting is the first thing cut if you fall behind |
| Feature pruning | Permutation importance on day 6 | After the race |
| Data download | Day 2 | Monday night, left running while you sleep |
| Final model | Frozen, scored on 2026 | Frozen settings, then refit on 2022 to the latest 2026 race and run on the Vegas grid |

The split is unchanged: train on 2022 to 2024, test on 2025, check on 2026 so far. Every season-based feature resets in round 1, which is what lets the model cope with the 2026 rule change.

## Day-by-day plan

Each day is a numpy lesson, the build tasks, and a checkpoint. Don't start the next day until the checkpoint passes; if you're behind, see the next section for what to cut.

### Mon 16 Nov: setup, first arrays, start the download

**numpy (30 min): an array is a list you can do maths on all at once.**

- [ ] `import numpy as np`, then make `grid = np.array([...])` and `finish = np.array([...])` by hand for the top 5 of any race
- [ ] Compute `finish - grid` in one line, with no loop. Negative means places gained
- [ ] Try `.shape`, `.dtype`, `.mean()`, `.max()` and `np.argmax()` on it
- [ ] Make a boolean mask: `gained = (finish - grid) < 0`, then `finish[gained]`

**Build:**

- [ ] Create the `f1-predictor` repo and folders, and turn on the FastF1 cache
- [ ] Load the 2025 Las Vegas race: `fastf1.get_session(2025, 'Las Vegas', 'R')`, then `session.load(laps=False, telemetry=False, weather=False, messages=False)`
- [ ] Pull `GridPosition` and `Position` out with `.to_numpy()` and redo the places-gained sum in numpy, then in pandas. The answers must match
- [ ] Write the download loop for 2022 to 2026: every race and every qualifying session, skipping 2026 events that haven't happened yet. Keep `laps=False`: qualifying results already carry Q1, Q2 and Q3 times
- [ ] Save one CSV per session as you go, so a crash halfway doesn't lose the rest. Start it before bed

**Checkpoint:** the 2025 Vegas places-gained table agrees in numpy and pandas, and the download is running.

### Tue 17 Nov: the dataset and the baseline

**numpy (25 min): blanks and the `axis` argument.**

- [ ] `np.nan` is a blank number. See why `np.mean([1, np.nan])` gives `nan` and `np.nanmean` doesn't
- [ ] Stack a driver's Q1, Q2 and Q3 times as a 2-D array and take `np.nanmin(..., axis=1)`. That is their best qualifying lap, one per row
- [ ] `np.where(finish <= 3, 1, 0)` builds a yes/no column in one line

**Build:**

- [ ] Check the download finished, then combine the per-session CSVs into `results.csv` and `quali.csv`
- [ ] Add best qualifying lap with `np.nanmin` across Q1 to Q3
- [ ] Add the podium target with `np.where`, and a classified flag from `ClassifiedPosition`
- [ ] Decide how grid 0 (pit-lane start) is treated. Setting it to 20 is a fine first rule
- [ ] Baseline on 2025: how often pole won, and how many of the grid top three made the podium, per race
- [ ] Same baseline on 2026 so far

**Checkpoint:** four baseline numbers (two measures, two seasons) in the README.

### Wed 18 Nov: qualifying and form features, without leakage

**numpy (30 min): whole-column maths and a rolling mean by hand.**

- [ ] For one race's best laps `t`: `(t - t.min()) / t.min() * 100` gives every gap to pole at once. That's broadcasting: one number applied to a whole array
- [ ] Take one driver's finishes for a season and build a 5-race average by hand with `np.cumsum`. Then check it against pandas `rolling(5).mean()`
- [ ] Shift that array by one place and see how it stops race N's result leaking into race N's feature

**Build:**

- [ ] Gap to pole and gap to teammate, with a fill rule for drivers with no time
- [ ] Sort by year and round, then build driver form: `groupby(['year', 'driver'])`, `shift(1)`, `rolling(5, min_periods=1).mean()`
- [ ] Team form, constructor points before the race, driver share of team points, reliability. All reset each season
- [ ] Fill round-1 blanks with the column median

**Checkpoint:** work two features out by hand for one driver and race, and they match the code. Then paste the feature code into Claude for a leakage review.

### Thu 19 Nov: logistic regression, by hand and with scikit-learn

**numpy (30 min): the model is one matrix sum.** This is the lesson to know cold for quant interviews.

- [ ] Scale by hand: `(X - X.mean(axis=0)) / X.std(axis=0)`, and check it against `StandardScaler` with `np.allclose`
- [ ] After fitting, rebuild the prediction yourself: `z = X @ model.coef_[0] + model.intercept_[0]`, then `p = 1 / (1 + np.exp(-z))`
- [ ] Confirm `np.allclose(p, model.predict_proba(X)[:, 1])` is `True`. If it is, you understand what the model does

**Build:**

- [ ] Split by time: train 2022 to 2024, test 2025. Never shuffle
- [ ] Fit `LogisticRegression` on the 8 features
- [ ] For each 2025 race, rank drivers by probability; the top one is the predicted winner
- [ ] Score winner hit rate and podium hits per race, next to the baseline
- [ ] Print the coefficients and say which features push the chance up or down

**Checkpoint:** a README table with baseline and logistic regression side by side on 2025.

### Fri 20 Nov: stronger model, 2026 check, freeze, dry run

**numpy (20 min): ranking and repeatability.**

- [ ] `order = np.argsort(-p)` sorts drivers from most to least likely; `drivers[order[:3]]` is your podium
- [ ] Set `random_state` on the model and see the score stay identical across runs

**Build:**

- [ ] Optional: Vegas history (average finish there in earlier years, blank for anyone with none)
- [ ] Fit `HistGradientBoostingClassifier` and score it against the baseline and logistic regression on 2025. Keep whichever wins
- [ ] Score the chosen model and the baseline on 2026 so far. Compare the lead over the baseline, not the raw scores
- [ ] Freeze: write the chosen model and its settings in the README. No tuning after this
- [ ] Refit with those settings on 2022 to the latest 2026 race
- [ ] Write the prediction step: load a qualifying session, build the 8 features for its 20 drivers from earlier races only, print the top 3 with probabilities
- [ ] Dry run it on the 2026 race before Vegas, where you already know the result

**Checkpoint:** the dry run goes from qualifying data to a printed top 3 without errors, and it's committed.

### Sat 21 Nov: the prediction

Qualifying ends around 13:00 Singapore time.

- [ ] Load the Vegas qualifying session. If FastF1 doesn't have it yet, type the 20 positions and best laps into a CSV from the official results
- [ ] Use qualifying order as the grid and adjust by hand for any announced grid penalties
- [ ] Run the prediction and sanity-check it: no driver with a blank feature at the top for the wrong reason
- [ ] Commit the top 3 and their probabilities to the README. The commit time proves it came before the race
- [ ] Explain-back: for each numpy line in the project, write one sentence on what it does, and ask Claude to correct you

**Checkpoint:** prediction pushed before Sun 22 Nov, 12:00 Singapore time.

### Sun 22 Nov: the race and the score

- [ ] Watch the race at 12:00
- [ ] Score the prediction: winner right or wrong, how many of the three made the podium, and what the grid baseline would have scored
- [ ] Add the result and one paragraph on what you'd change to the README

One race is a tiny sample, so the 2025 and 2026 scores are the real test of the model. Vegas is the fun one.

## If you fall behind

The one thing that can't slip is Friday's dry run: without it, Saturday becomes debugging under a deadline. Cut in this order, and only as far as you need to:

1. Vegas history feature (Friday)
2. Gradient boosting: predict with logistic regression instead (Friday)
3. Driver share of team points and reliability (Wednesday)
4. Gap to teammate (Wednesday)
5. The 2026 check: still freeze, just score it after the race (Friday)

With only grid, gap to pole, driver form, team form and constructor points, logistic regression still makes a real prediction. Never cut the numpy lessons or the leakage check: those are the parts you're here to learn.

## numpy cheat sheet

| Code | What it does | Day |
| --- | --- | --- |
| `np.array([...])`, `.to_numpy()` | Make an array, or pull one out of a pandas column | Mon |
| `a - b`, `a * 100` | Maths on every element at once, no loop | Mon |
| `.shape`, `.dtype` | Size and type of an array | Mon |
| `a[a < 0]` | Keep only elements that pass a test (boolean mask) | Mon |
| `np.argmax(a)` | Position of the largest value | Mon |
| `np.nan`, `np.isnan`, `np.nanmin`, `np.nanmean` | Blanks, and maths that skips them | Tue |
| `axis=0` / `axis=1` | Work down columns / across rows | Tue |
| `np.where(cond, x, y)` | Pick x where the test is true, y elsewhere | Tue |
| `np.cumsum(a)` | Running total; the basis of a rolling average | Wed |
| `X @ w` | Matrix times vector: every driver's score at once | Thu |
| `np.exp`, `1 / (1 + np.exp(-z))` | The sigmoid that turns a score into a probability | Thu |
| `np.allclose(a, b)` | Check two arrays match, allowing rounding | Thu |
| `np.argsort(-p)` | Order from highest to lowest | Fri |
