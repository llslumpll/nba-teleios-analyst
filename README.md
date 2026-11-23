# NBA Teleios Analyst

A personal NBA analytics lab. It predicts every game's winner and margin, freezes each prediction before tip-off, grades it after the game, and keeps score against the betting market. It is built on the foundation of [Prop Edge NBA](https://github.com/llslumpll/nba-prop-board): the same free data sources, and the same freeze-and-grade honesty. It runs free on GitHub Actions, so no computer needs to be on.

**Status:** step 1 (data) and step 2 (game-winner model) are live. Step 3, the daily predictions and grading, is wired. The analyst chat comes next.

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
