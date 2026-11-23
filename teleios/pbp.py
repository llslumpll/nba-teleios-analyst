"""
Shot data from ESPN play-by-play (sportsdataverse, free). Every field-goal
attempt gets a place on the court and a type, then two compact tables are
kept in data/cache/ (the raw play-by-play, 4-25 MB a season, is thrown away):

  shots_{season}.parquet       one row per field-goal attempt
      game_id, date, team, opp, pid, period, secs_left, zone, family,
      dist, made, pts, assisted, xpts
  team_shots_{season}.parquet  one row per team per game
      attempts / makes / points by zone, xPTS (what a league-average
      shooter scores on those same shots), shot-making luck = pts - xpts

Zones: rim (<4 ft), paint (4-13 ft), mid (14 ft+ twos), corner3, arc3
(above the break). Heaves (36 ft+) are kept but excluded from luck.
Families: dunk, layup, hook, jumper, pullup, stepback, fadeaway, tip, other.

xPTS uses league points per attempt for each zone x family in that
season. (Within-season league averages are very stable; this is the only
place a season's own totals are used.)
"""

from __future__ import annotations

import polars as pl
import requests

from teleios import data, teams

PBP_URL = data.BASE + "/espn_nba_pbp/play_by_play_{s}.parquet"
SHOTS = "shots_{s}.parquet"
TEAM_SHOTS = "team_shots_{s}.parquet"
ZONES = ("rim", "paint", "mid", "corner3", "arc3")

FAMILY = [  # first match wins
    ("tip", "(?i)tip|putback"), ("dunk", "(?i)dunk|alley"), ("hook", "(?i)hook"),
    ("stepback", "(?i)step ?back"), ("fadeaway", "(?i)fade|turnaround"), ("pullup", "(?i)pull ?up"),
    ("layup", "(?i)layup|finger roll|floating|floater|reverse"), ("jumper", "(?i)jump"),
]


def _team_code(col: str) -> pl.Expr:
    return pl.col(col).map_elements(lambda x: teams.canon(x, "espn"), return_dtype=pl.Utf8)


def summarize(pbp: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Older seasons (2023-24) leave points_attempted empty: rebuild it from
    # the play text and the shot's distance from the rim.
    ax, ay = pl.col("coordinate_x").abs(), pl.col("coordinate_y")
    d_txt = pl.col("text").str.extract(r"(\d+)-foot", 1).cast(pl.Float64)
    d_xy = ((ax - 41.75) ** 2 + ay ** 2).sqrt()
    d0 = pl.coalesce(d_txt, d_xy)
    derived = (pl.when(pl.col("type_text").str.contains("(?i)free throw")).then(pl.lit(1))
               .when(pl.col("text").str.contains("(?i)three point")).then(pl.lit(3))
               .when((ay.abs() >= 21.5) & (d0 >= 21.5)).then(pl.lit(3))
               .when(d0 >= 23.5).then(pl.lit(3))
               .otherwise(pl.lit(2)))
    pbp = pbp.with_columns(points_attempted=pl.when(pl.col("shooting_play"))
                           .then(pl.coalesce(pl.col("points_attempted").cast(pl.Int32), derived.cast(pl.Int32)))
                           .otherwise(pl.col("points_attempted")))
    sh = pbp.filter(pl.col("shooting_play") & pl.col("points_attempted").is_in([2, 3])
                    & pl.col("athlete_id_1").is_not_null())
    x, y = pl.col("coordinate_x").abs(), pl.col("coordinate_y")
    coord_dist = ((x - 41.75) ** 2 + y ** 2).sqrt()
    fam = pl.lit("other")
    for name, pat in reversed(FAMILY):
        fam = pl.when(pl.col("type_text").str.contains(pat)).then(pl.lit(name)).otherwise(fam)
    sh = sh.with_columns(
        dist=pl.coalesce(pl.col("text").str.extract(r"(\d+)-foot", 1).cast(pl.Float64), coord_dist).round(0),
        is_home=pl.col("team_id") == pl.col("home_team_id"),
        family=fam,
    ).with_columns(
        zone=pl.when(pl.col("points_attempted") == 3)
        .then(pl.when(pl.col("coordinate_y").abs() >= 21.5).then(pl.lit("corner3")).otherwise(pl.lit("arc3")))
        .when(pl.col("dist") < 4).then(pl.lit("rim"))
        .when(pl.col("dist") < 14).then(pl.lit("paint"))
        .otherwise(pl.lit("mid")),
        team=pl.when(pl.col("is_home")).then(pl.col("home_team_abbrev")).otherwise(pl.col("away_team_abbrev")),
        opp=pl.when(pl.col("is_home")).then(pl.col("away_team_abbrev")).otherwise(pl.col("home_team_abbrev")),
        made=pl.col("scoring_play").fill_null(False),
    )
    shots = sh.select(
        pl.col("game_id").cast(pl.Int64), pl.col("game_date").alias("date"), pl.col("season").cast(pl.Int32),
        pl.col("season_type").cast(pl.Int32),
        _team_code("team").alias("team"), _team_code("opp").alias("opp"),
        pl.col("athlete_id_1").cast(pl.Int64).alias("pid"),
        pl.col("period_number").cast(pl.Int32).alias("period"),
        pl.col("end_game_seconds_remaining").cast(pl.Int32).alias("secs_left"),
        (pl.col("home_score") - pl.col("away_score")).cast(pl.Int32).alias("home_lead_after"),
        "zone", "family", pl.col("dist").cast(pl.Int32),
        "made", (pl.col("made").cast(pl.Int32) * pl.col("points_attempted")).cast(pl.Int32).alias("pts"),
        (pl.col("made") & pl.col("athlete_id_2").is_not_null()).alias("assisted"),
        pl.col("points_attempted").cast(pl.Int32).alias("value"),
    ).filter(pl.col("team").is_not_null() & pl.col("opp").is_not_null())
    heave = pl.col("dist") >= 36
    rates = shots.filter(~heave).group_by(["zone", "family"]).agg(ppa=pl.col("pts").mean(), n=pl.len())
    zone_rates = shots.filter(~heave).group_by("zone").agg(zppa=pl.col("pts").mean())
    shots = shots.join(rates, on=["zone", "family"], how="left").join(zone_rates, on="zone", how="left")
    shots = shots.with_columns(
        xpts=pl.when(heave).then(pl.col("pts").cast(pl.Float64))   # heaves: no luck either way
        .when(pl.col("n") >= 200).then(pl.col("ppa")).otherwise(pl.col("zppa"))
    ).drop(["ppa", "n", "zppa"])

    agg = [pl.len().alias("fga"), pl.col("pts").sum().alias("fg_pts"), pl.col("xpts").sum().alias("fg_xpts"),
           pl.col("assisted").sum().alias("assisted")]
    for z in ZONES:
        agg += [(pl.col("zone") == z).sum().alias(f"{z}_fga"),
                ((pl.col("zone") == z) & pl.col("made")).sum().alias(f"{z}_fgm")]
    ts = shots.group_by(["game_id", "date", "season", "team", "opp"]).agg(agg).with_columns(
        luck=pl.col("fg_pts") - pl.col("fg_xpts"))
    return shots, ts.sort(["date", "game_id", "team"])


def refresh(seasons=None) -> list[dict]:
    """Rebuild the shot tables for the current season (and any season that
    has none yet). Never raises; keeps old tables on failure."""
    seasons = seasons or data.SEASONS
    out = []
    for s in seasons:
        dest = data.CACHE / TEAM_SHOTS.format(s=s)
        if dest.exists() and s != data.CURRENT_SEASON:
            continue
        rec = {"season": s, "checked_at": data.now_utc()}
        try:
            r = requests.get(PBP_URL.format(s=s), timeout=180)
            r.raise_for_status()
            tmp = data.CACHE / f"pbp_{s}.tmp"
            tmp.write_bytes(r.content)
            shots, ts = summarize(pl.read_parquet(tmp))
            tmp.unlink()
            shots.write_parquet(data.CACHE / SHOTS.format(s=s))
            ts.write_parquet(dest)
            rec.update(state="ok", shots=shots.height, games=ts["game_id"].n_unique())
        except Exception as e:  # noqa: BLE001
            rec.update(state="failed (kept old copy)" if dest.exists() else "failed", error=str(e)[:200])
        out.append(rec)
    return out


def team_shots() -> pl.DataFrame:
    frames = [pl.read_parquet(p) for s in data.SEASONS if (p := data.CACHE / TEAM_SHOTS.format(s=s)).exists()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def shots() -> pl.DataFrame:
    frames = [pl.read_parquet(p) for s in data.SEASONS if (p := data.CACHE / SHOTS.format(s=s)).exists()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def luck_by_game() -> dict[int, dict[str, float]]:
    """game_id -> {team: shot-making luck (FG points above a league-average
    shooter on the same shots)}."""
    ts = team_shots()
    out: dict[int, dict[str, float]] = {}
    if ts.is_empty():
        return out
    for gid, t, l in ts.select(["game_id", "team", "luck"]).iter_rows():
        out.setdefault(gid, {})[t] = float(l)
    return out
