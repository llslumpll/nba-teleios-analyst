"""
Teleios Tracker: pulls NBA.com player-tracking stats (from the arena camera
system) once a day and saves them for NBA Teleios Analyst.

Runs on this Mac because NBA.com blocks cloud servers. Reads public stats
pages only; needs no account, password or key. Everything it writes stays
inside this folder:

  latest/<name>.csv              today's season-to-date numbers
  snapshots/<date>/<name>.csv.gz one copy per day, so changes can be tracked
  status.json                    what worked, what didn't, when
  logs/                          output of each run

Uninstall any time with uninstall.command.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
LATEST = HERE / "latest"
SNAP = HERE / "snapshots"
STATUS = HERE / "status.json"


def season_label(now: datetime) -> str:
    y = now.year if now.month >= 9 else now.year - 1
    return f"{y}-{str(y + 1)[-2:]}"


# NBA.com "tracking" measure types (stats.nba.com/stats/leaguedashptstats)
MEASURES = {
    "speed_distance": "SpeedDistance",
    "drives": "Drives",
    "catch_shoot": "CatchShoot",
    "pullup": "PullUpShot",
    "passing": "Passing",
    "touches": "Possessions",
    "rebounding": "Rebounding",
    "rim_defense": "Defense",
    "efficiency": "Efficiency",
    "paint_touch": "PaintTouch",
    "post_touch": "PostTouch",
    "elbow_touch": "ElbowTouch",
}
DEFENDER_RANGES = {
    "very_tight_0_2ft": "0-2 Feet - Very Tight",
    "tight_2_4ft": "2-4 Feet - Tight",
    "open_4_6ft": "4-6 Feet - Open",
    "wide_open_6ft_plus": "6+ Feet - Wide Open",
}
PAUSE = 2.0  # seconds between requests: polite to NBA.com


def fetch(make, tries: int = 3):
    last = None
    for i in range(tries):
        try:
            df = make().get_data_frames()[0]
            return df
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(5 * (i + 1))
    raise last


def main() -> int:
    from nba_api.stats.endpoints import leaguedashplayerptshot, leaguedashptstats, leaguedashteamptshot

    now = datetime.now()
    season = season_label(now)
    day = now.strftime("%Y-%m-%d")
    LATEST.mkdir(exist_ok=True)
    (SNAP / day).mkdir(parents=True, exist_ok=True)
    jobs = []
    for name, m in MEASURES.items():
        for who in ("Player", "Team"):
            jobs.append((f"{who.lower()}_{name}", lambda m=m, who=who: leaguedashptstats.LeagueDashPtStats(
                season=season, season_type_all_star="Regular Season", per_mode_simple="PerGame",
                player_or_team=who, pt_measure_type=m, timeout=60)))
    for name, rng in DEFENDER_RANGES.items():
        jobs.append((f"player_shots_{name}", lambda rng=rng: leaguedashplayerptshot.LeagueDashPlayerPtShot(
            season=season, season_type_all_star="Regular Season", per_mode_simple="Totals",
            close_def_dist_range_nullable=rng, timeout=60)))
        jobs.append((f"team_shots_{name}", lambda rng=rng: leaguedashteamptshot.LeagueDashTeamPtShot(
            season=season, season_type_all_star="Regular Season", per_mode_simple="Totals",
            close_def_dist_range_nullable=rng, timeout=60)))

    status = {"run_at": now.strftime("%Y-%m-%d %H:%M"), "season": season, "ok": {}, "failed": {}}
    for name, make in jobs:
        try:
            df = fetch(make)
            df.to_csv(LATEST / f"{name}.csv", index=False)
            with open(LATEST / f"{name}.csv", "rb") as src, gzip.open(SNAP / day / f"{name}.csv.gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            status["ok"][name] = len(df)
            print(f"ok      {name}: {len(df)} rows", flush=True)
        except Exception as e:  # noqa: BLE001
            status["failed"][name] = f"{type(e).__name__}: {str(e)[:150]}"
            print(f"FAILED  {name}: {status['failed'][name]}", flush=True)
        time.sleep(PAUSE)
    STATUS.write_text(json.dumps(status, indent=1))
    print(f"\n{len(status['ok'])} of {len(jobs)} tables saved.", flush=True)
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
