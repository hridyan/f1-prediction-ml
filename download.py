"""Download every F1 session from 2022 to today into data/raw/, one CSV per session.

Run from the repo root:   python download.py
Safe to stop and rerun: sessions already saved are skipped.

Order: all qualifying, sprint and race results first (fast, and all the roadmap
needs), then practice and sprint qualifying (slow). FastF1 allows 500 requests
an hour, so when that limit is hit the script waits and carries on by itself.
"""
import time
from pathlib import Path

import fastf1
import pandas as pd
from fastf1.exceptions import FastF1CriticalError, RateLimitExceededError

ROOT = Path(__file__).parent
OUT = ROOT / 'data' / 'raw'
YEARS = range(2022, 2027)
WAIT_MINUTES = 10            # pause after hitting the hourly request limit

# Full session name in the schedule -> short code used in file names
CODES = {
    'Practice 1': 'FP1',
    'Practice 2': 'FP2',
    'Practice 3': 'FP3',
    'Sprint Qualifying': 'SQ',   # 2024 onwards
    'Sprint Shootout': 'SS',     # 2023 only
    'Sprint': 'S',
    'Qualifying': 'Q',
    'Race': 'R',
}
# These sessions are only in F1's live timing (needs the F1 TV login), and their
# results are worked out from lap times, so they need laps and race control messages.
NEEDS_LAPS = {'FP1', 'FP2', 'FP3', 'SQ', 'SS'}

fastf1.set_log_level('ERROR')
fastf1.Cache.configure(cache_dir=str(ROOT / 'cache'))
OUT.mkdir(parents=True, exist_ok=True)

now = pd.Timestamp.now(tz='UTC').tz_localize(None)   # schedule times are naive UTC

# Step 1: list every session that has already taken place
jobs = []
for year in YEARS:
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    for _, event in schedule.iterrows():
        for i in range(1, 6):
            name = event[f'Session{i}']
            code = CODES.get(name)
            if code is None:
                continue                                  # blank slot or unknown session
            start = event[f'Session{i}DateUtc']
            if pd.isna(start) or start > now:
                continue                                  # hasn't happened yet
            jobs.append({'year': year, 'rnd': event['RoundNumber'], 'name': name,
                         'code': code, 'event': event['EventName']})

# Step 2: quick sessions (Q, S, R) first, slow ones (practice, SQ, SS) after.
# False sorts before True, and the sort keeps the date order within each group.
jobs.sort(key=lambda job: job['code'] in NEEDS_LAPS)

# Step 3: download
saved = skipped = empty = failed = 0

for n, job in enumerate(jobs, start=1):
    year, rnd, code = job['year'], job['rnd'], job['code']
    path = OUT / f'{year}_{rnd:02d}_{code}.csv'
    if path.exists():
        skipped += 1
        continue

    label = f"[{n}/{len(jobs)}] {year} R{rnd:02d} {code:<3} {job['event']}"
    full = code in NEEDS_LAPS

    while True:                                           # repeat only after a rate-limit wait
        try:
            session = fastf1.get_session(year, rnd, job['name'])
            session.load(laps=full, telemetry=False, weather=False, messages=full)

            df = session.results.copy()
            if df.empty or df['Position'].isna().all():
                # Don't save: a later rerun will try this session again
                print(f'EMPTY   {label}')
                empty += 1
            else:
                df['Year'] = year
                df['Round'] = rnd
                df['EventName'] = job['event']
                df['Session'] = code
                df.to_csv(path, index=False)
                print(f'saved   {label}')
                saved += 1
            break

        except RateLimitExceededError:
            print(f'{time.strftime("%H:%M")} hourly request limit reached, '
                  f'waiting {WAIT_MINUTES} min...')
            time.sleep(WAIT_MINUTES * 60)                 # then try the same session again
        except FastF1CriticalError as e:
            print(f'STOPPED {label}: {e}')
            raise SystemExit(1)
        except Exception as e:
            print(f'FAILED  {label}: {type(e).__name__}: {e}')
            failed += 1
            break

print(f'\nDone. saved {saved}, already had {skipped}, empty {empty}, failed {failed}')
