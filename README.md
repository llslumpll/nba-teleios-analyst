# NBA Teleios Analyst

A personal NBA analytics lab. It predicts every game's winner and margin, freezes each prediction before tip-off, grades it after the game, and keeps score against the betting market. It is built on the foundation of [Prop Edge NBA](https://github.com/llslumpll/nba-prop-board): the same free data sources, and the same freeze-and-grade honesty. It runs free on GitHub Actions, so no computer needs to be on.

**Status:** game predictions, player stat-line projections, daily grading, the scorecard, the patterns notebook, shot data, NBA.com tracking stats (from the Mac tracker), the analyst chat and the monthly retune are all live.

## How it works

| Step | What happens | Where |
|---|---|---|
| Data | ESPN schedules and box scores (sportsdataverse), ESPN injuries and rosters, Kalshi game odds | `teleios/data.py`, `teleios/sources.py` |
| Model | A team's strength is the sum of the players who will actually play, plus home court, rest, travel and altitude | `teleios/model.py` |
| Daily run | Refresh, grade, re-fit, predict today and tomorrow, lock at tip-off | `teleios/daily.py`, `.github/workflows/daily.yml` |
| Scorecard | Accuracy, Brier score, log loss, calibration, and Teleios vs the market | `teleios/scorecard.py` → `data/scorecard.json` |
| Notebook | Repeated misses that hold up in both halves of the games | `teleios/notebook.py` → `data/notebook.json` |

## The game model

The model predicts the home team's margin:

```
margin = home court + rest/back-to-back + travel + altitude
       + team term (coaching/system, shrunk toward 0)
       + Σ home players (projected minutes / 48 × player rating)
       − Σ away players (same)
P(home wins) = Φ(margin / σ)
```

- **Player ratings** come from every past game's final margin, using the minutes each player actually played. This is a game-level regularised plus-minus.
  - Each rating is pulled toward a **box-score prior**: what a player's per-48 points, rebounds, assists, steals, blocks, turnovers, fouls and shot volume usually mean for winning. The prior is re-learned every July 1 from past seasons only.
  - Older games fade with a half-life.
- **Injuries, rest days and trades move the prediction automatically,** because tonight's strength is the sum of tonight's available players.
  - A player listed Out or Doubtful is removed, and his minutes go to his teammates.
  - A player listed Questionable keeps 65% of his projected minutes.
- **Settings** were chosen by walk-forward tuning on 2024-25 (`data/model_params.json`):
  - half-life 240 days
  - player pull 12
  - team pull 6
  - σ = 12.75

## Results so far

**Backtest on 2025-26.** The season was never seen while tuning, and the model knew who played but not their minutes.

| Model | Games | Picked winner | Brier | Log loss |
|---|---|---|---|---|
| **Teleios** | 1,322 | **70.0%** | **0.202** | **0.589** |
| Elo, FiveThirtyEight style | 1,322 | 67.4% | 0.211 | 0.612 |
| Always the home team | 1,322 | 55.5% | — | — |

On calibration, when Teleios said 80–90%, the favourite won 84.8% of the time.

**2026-27 so far** (backfill, 245 games with a market price):

| | Picked winner | Brier |
|---|---|---|
| Teleios | 67.4% | 0.198 |
| FanDuel closing line | 70.2% | 0.193 |
| Prop Edge (partly blended with market) | 69.4% | 0.194 |

- The market is ahead so far, but the gap is not yet statistically clear (z = 1.1).
- Most of the gap is October, when offseason roster changes made old player data less reliable: 62.4% vs 68.2%. In November it was 70.0% vs 71.3%.

## The player model

Built in `teleios/players.py`, with settings in `data/player_params.json`. Each line is "if he plays". A player who sits is void, not a miss.

- **Minutes:** recency-weighted minutes (half-life of 3 games), scaled toward 240 team minutes among the players who will actually play, capped at 42. Starters lose minutes when the game model expects a blowout.
- **Rates:** each stat's per-minute rate is recency-weighted, with a lighter carry-over from last season, and shrunk toward the league rate for his minutes role.
- **Context:** how much the opponent allows of each stat (strength 0.6). Pace was tested and switched off.
- **Ranges:** 50% and 80% ranges, calibrated on the tuning season.
  - On 2025-26, the 80% range held 79.8% of points results and 80.3% of rebounds results.
  - Steal and block ranges are rough, because those stats are mostly 0s and 1s.

**Backtest:** tuned on 2024-25, tested on 2025-26. The sample is 16,164 regular-season player-games projected for 20+ minutes. Numbers are average miss.

| | Points | Rebounds | Assists | Threes | PRA |
|---|---|---|---|---|---|
| **Teleios** | **5.09** | **1.99** | **1.54** | **1.03** | **6.40** |
| Prop Edge (published) | 5.10 | 2.00 | 1.54 | 1.04 | 6.42 |
| Season average so far | 5.36 | 2.06 | 1.59 | 1.05 | 6.85 |

## Learning

- **Monthly retune** (`teleios/retune.py`, `retune.yml`, the 1st of each month): re-fits the game and player settings on recent games. New settings are adopted only if they beat the current ones on the latest 30% of this season, which the search never saw.
  - It skips itself until there are 300 held-out games.
  - Every run is logged in `data/retune_log.json` with the old settings, so it can be undone.
- **Patterns notebook:** repeated misses that hold up in both halves of the games.
- **Experiments:** ideas tested before they are trusted, kept in `data/experiments.json`. For example, the shot-luck adjustment was tested and rejected.

## Honesty rules

- A prediction updates until tip-off, then it is locked forever.
- Live predictions and the backfill are always scored separately. The backfill was made after the fact.
- A failed source keeps its last good copy, and the error is logged in `data/run_log.json`.
- Nothing is invented.

## Running it by hand

```
pip install -r requirements.txt
python -m teleios.daily              # full daily run
python -m teleios.daily --no-fetch   # cached data only
python -m teleios.backtest           # re-tune (slow, ~40 min)
python -m teleios.backfill           # rebuild the 2026-27 backfill
```

## The analyst chat

The chat is a Claude artifact page (`chat/index.html`). It reads `chat/data.json`, which `python -m teleios.bundle` builds from everything in `data/`. A scheduled Claude task rebuilds that file and republishes it twice a day.

The analyst answers only through its tools:
- tonight's games and why each pick was made
- team and player reports
- rankings and leaders
- head-to-head results
- injuries
- the scorecard
- the notebook and experiments
