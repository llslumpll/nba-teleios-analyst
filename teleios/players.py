"""
Player stat-line model: projects points, rebounds, assists, threes,
steals, blocks, turnovers and PRA for every player in a game, with a 50%
and an 80% range, using only games before that date.

  minutes   recency-weighted minutes over his recent games (half-life in
            games), scaled toward 240 team minutes with everyone who is
            actually playing (so absences free minutes for teammates),
            capped at 42; starters lose minutes when the game model expects
            a blowout
  rates     per-minute rate for each stat, recency-weighted, shrunk toward
            the league rate of players in a similar minutes role (shrink
            strength k, in minutes)
  context   game pace vs his team's usual pace; how much the opponent allows
            of that stat (each with a tuned strength; 0 = off)
  ranges    variance = phi*mu + kappa*mu^2 per stat, fit on the tuning
            season; points and PRA use a normal curve, the rest a negative
            binomial

Season labels are the year a season ends (2026-27 is 2027).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np
import polars as pl
from scipy import stats as sst

from teleios import data

STATS = ("pts", "reb", "ast", "fg3m", "stl", "blk", "tov")
COMBOS = {"pra": ("pts", "reb", "ast")}
NB_STATS = ("reb", "ast", "fg3m", "stl", "blk", "tov")
TIERS = (0.0, 18.0, 28.0, 99.0)  # minutes tiers for the prior


@dataclass
class PParams:
    min_half_life: float = 6.0          # games
    season_carry: float = 0.7           # weight kept across a season break
    team_pull: float = 0.7              # how far minutes move toward 240 team minutes
    blowout_cut: float = 0.15           # starter minutes lost per point of expected margin beyond 8
    pace_beta: float = 1.0
    def_beta: float = 0.0
    team_half_life: float = 12.0        # games, for pace and opponent allowances
    rate_half_life: dict = field(default_factory=lambda: {s: 25.0 for s in STATS})
    shrink_k: dict = field(default_factory=lambda: {s: 300.0 for s in STATS})
    var: dict = field(default_factory=dict)  # stat -> [phi, kappa]

    def to_dict(self):
        return asdict(self)


def _played(pg: pl.DataFrame, games: pl.DataFrame) -> pl.DataFrame:
    g = games.select(["game_id", "tip_utc", "season", "season_type", "home", "away"])
    p = pg.filter(~pl.col("dnp") & (pl.col("min") > 0)).join(g, on="game_id", how="inner")
    p = p.with_columns(home_flag=(pl.col("team") == pl.col("home")),
                       opp=pl.when(pl.col("team") == pl.col("home")).then(pl.col("away")).otherwise(pl.col("home")))
    return p.sort(["tip_utc", "game_id"])


def _ewm_pre(keys: np.ndarray, seasons: np.ndarray, vals: np.ndarray, half_life: float, carry: float):
    """For rows ordered in time, the weighted sum of each key's EARLIER rows
    (decay per row of that key, extra decay `carry` at a season change).
    vals: (n, m). Returns sums (n, m) and weight totals (n,)."""
    a = 0.5 ** (1.0 / half_life)
    n, m = vals.shape
    out = np.zeros((n, m))
    wtot = np.zeros(n)
    state: dict = {}
    for i in range(n):
        k = keys[i]
        st = state.get(k)
        if st is None:
            st = [np.zeros(m), 0.0, seasons[i]]
            state[k] = st
        if st[2] != seasons[i]:
            st[0] *= carry
            st[1] *= carry
            st[2] = seasons[i]
        out[i] = st[0]
        wtot[i] = st[1]
        st[0] = st[0] * a + vals[i]
        st[1] = st[1] * a + 1.0
    return out, wtot


class PlayerModel:
    def __init__(self, games: pl.DataFrame, pg: pl.DataFrame, params: PParams | None = None):
        self.p = params or PParams()
        self.games = games
        self.df = _played(pg, games)
        d = self.df
        self.n = d.height
        self.pid = d["pid"].to_numpy()
        self.season = d["season"].to_numpy()
        self.mins = d["min"].to_numpy()
        self.Y = np.column_stack([d[s].to_numpy() for s in STATS])
        self.gid = d["game_id"].to_numpy()
        self.team = d["team"].to_numpy()
        self.opp = d["opp"].to_numpy()
        self._team_context()

    # ---- team pace and what each defence allows, before each game ----------
    def _team_context(self):
        d = self.df
        tg = d.group_by(["game_id", "tip_utc", "season", "team", "opp"]).agg(
            poss=(pl.col("fga") + 0.44 * pl.col("fta") - pl.col("oreb") + pl.col("tov")).sum(),
            tmin=pl.col("min").sum(),
            **{s: pl.col(s).sum() for s in STATS},
        ).sort(["tip_utc", "game_id"])
        tg = tg.with_columns(pace=pl.col("poss") * 48 * 5 / pl.col("tmin").clip(200, 400))
        # what the OPPONENT of each row allowed = this team's own production, keyed by the defending team
        h = self.p.team_half_life
        keys = tg["team"].to_numpy()
        seas = tg["season"].to_numpy()
        pace_s, pace_w = _ewm_pre(keys, seas, tg["pace"].to_numpy()[:, None], h, 0.5)
        dkeys = tg["opp"].to_numpy()
        allowed = np.column_stack([tg[s].to_numpy() * 240.0 / tg["tmin"].clip(200, 400).to_numpy() for s in STATS])
        al_s, al_w = _ewm_pre(dkeys, seas, allowed, h, 0.5)
        tg = tg.with_columns(
            pace_pre=pl.Series(np.where(pace_w > 0, pace_s[:, 0] / np.maximum(pace_w, 1e-9), np.nan)),
        )
        for j, s in enumerate(STATS):
            tg = tg.with_columns(pl.Series(f"allow_{s}", np.where(al_w > 0, al_s[:, j] / np.maximum(al_w, 1e-9), np.nan)))
        # league averages per season (centring only)
        lg = tg.group_by("season").agg(pl.col("pace").mean().alias("lg_pace"),
                                       *[(pl.col(s) * 240.0 / pl.col("tmin").clip(200, 400)).mean().alias(f"lg_{s}") for s in STATS])
        self.tg = tg.join(lg, on="season", how="left")
        own = self.tg.select(["game_id", "team", "pace_pre", "lg_pace"])
        opp = self.tg.select(["game_id", pl.col("team").alias("opp"), pl.col("pace_pre").alias("opp_pace_pre"),
                              *[pl.col(f"allow_{s}").alias(f"opp_allow_{s}") for s in STATS],
                              *[pl.col(f"lg_{s}") for s in STATS]])
        # opp_allow_s is keyed by defending team: rebuild keyed by defender
        defn = self.tg.select(["game_id", pl.col("opp").alias("defender"), *[f"allow_{s}" for s in STATS]])
        x = self.df.select(["game_id", "team", "opp"]).with_row_index("row")
        x = x.join(own, on=["game_id", "team"], how="left")
        x = x.join(opp.select(["game_id", "opp", "opp_pace_pre", *[f"lg_{s}" for s in STATS]]), on=["game_id", "opp"], how="left")
        # opponent's defence: rows of tg where tg.opp == this opp, i.e. the defender column equals our opp,
        # for the same game, that row describes OUR team's production, so use the pre-game allowance of `opp`
        defn = defn.rename({"defender": "opp"})
        x = x.join(defn, on=["game_id", "opp"], how="left").sort("row")
        self.ctx = x

    # ---- rates and minutes before each row ---------------------------------
    def _priors(self):
        """League per-minute rate by minutes tier from the PREVIOUS season."""
        d = self.df.with_columns(season_prev=pl.col("season"))
        pm = d.group_by(["season", "pid"]).agg(m=pl.col("min").sum(), g=pl.len(), **{s: pl.col(s).sum() for s in STATS})
        pm = pm.with_columns(mpg=pl.col("m") / pl.col("g"))
        out = {}
        for season in sorted(set(self.season.tolist())):
            prev = pm.filter(pl.col("season") == season - 1)
            if prev.is_empty():
                prev = pm.filter(pl.col("season") == season)
            tiers = []
            for lo, hi in zip(TIERS[:-1], TIERS[1:]):
                t = prev.filter((pl.col("mpg") >= lo) & (pl.col("mpg") < hi))
                tiers.append(np.array([t[s].sum() / max(t["m"].sum(), 1) for s in STATS]))
            out[season] = tiers
        return out

    def project_rows(self, rows: np.ndarray | None = None, margins: dict | None = None) -> dict:
        """Projections for every played row (or the given row indexes), using
        the minutes context of everyone who played in that game."""
        p = self.p
        # minutes
        ms, mw = _ewm_pre(self.pid, self.season, self.mins[:, None], p.min_half_life, p.season_carry)
        base_min = np.where(mw > 0.5, ms[:, 0] / np.maximum(mw, 1e-9), 14.0)
        base_min = np.where(mw < 2.5, 0.5 * base_min + 0.5 * 18.0, base_min)
        # team scaling among the players who played
        dfm = pl.DataFrame({"gid": self.gid, "team": self.team, "b": base_min})
        tot = dfm.group_by(["gid", "team"]).agg(pl.col("b").sum().alias("tot"))
        tot_map = {(g, t): v for g, t, v in tot.iter_rows()}
        team_tot = np.array([tot_map[(g, t)] for g, t in zip(self.gid, self.team)])
        scale = 1.0 + p.team_pull * (240.0 / np.maximum(team_tot, 60.0) - 1.0)
        proj_min = np.minimum(base_min * scale, 42.0)
        if margins and p.blowout_cut:
            home = self.df["home_flag"].to_numpy()
            m = np.array([abs(margins.get(str(g), 0.0)) for g in self.gid])
            cut = np.minimum(p.blowout_cut * np.maximum(m - 8.0, 0.0), 3.0)
            proj_min = np.where(base_min >= 28.0, proj_min - cut, proj_min)
        # rates
        priors = self._priors()
        tier_idx = np.digitize(base_min, TIERS[1:-1])
        prior = np.array([priors[s][t] for s, t in zip(self.season, tier_idx)])
        mu = np.zeros((self.n, len(STATS)))
        for j, s in enumerate(STATS):
            hl = p.rate_half_life[s]
            vals = np.column_stack([self.Y[:, j], self.mins])
            sums, _w = _ewm_pre(self.pid, self.season, vals, hl, p.season_carry)
            k = p.shrink_k[s]
            rate = (sums[:, 0] + k * prior[:, j]) / (sums[:, 1] + k)
            mu[:, j] = proj_min * rate
        # context
        c = self.ctx
        if p.pace_beta:
            tp, op, lp = c["pace_pre"].to_numpy(), c["opp_pace_pre"].to_numpy(), c["lg_pace"].to_numpy()
            tp = np.where(np.isnan(tp), lp, tp)
            op = np.where(np.isnan(op), lp, op)
            pf = ((tp + op) / 2.0) / tp
            mu = mu * (pf ** p.pace_beta)[:, None]
        if p.def_beta:
            for j, s in enumerate(STATS):
                al, lg = c[f"allow_{s}"].to_numpy(), c[f"lg_{s}"].to_numpy()
                f = np.where(np.isnan(al) | (lg <= 0), 1.0, al / np.maximum(lg, 1e-9))
                mu[:, j] *= np.clip(f, 0.75, 1.25) ** p.def_beta
        return {"min": proj_min, "base_min": base_min, "mu": mu}


# ---------------------------------------------------------------------------
# ranges

def fit_var(mu: np.ndarray, y: np.ndarray) -> list[float]:
    r2 = (y - mu) ** 2
    X = np.column_stack([mu, mu ** 2])
    coef, *_ = np.linalg.lstsq(X, r2, rcond=None)
    return [float(max(coef[0], 0.3)), float(max(coef[1], 0.0))]


def _v(mu, var):
    phi, kappa = var[0], var[1]
    scale = var[2] if len(var) > 2 else 1.0
    return max((phi * mu + kappa * mu * mu) * scale, 1e-6)


def interval(stat: str, mu: float, var: list[float], q: float) -> tuple[float, float]:
    """Central q range for one projection. var = [phi, kappa, scale]."""
    v = _v(mu, var)
    lo_p, hi_p = (1 - q) / 2, 1 - (1 - q) / 2
    if stat in NB_STATS and v > mu > 0:
        r = mu * mu / (v - mu)
        pr = r / (r + mu)
        return float(sst.nbinom.ppf(lo_p, r, pr)), float(sst.nbinom.ppf(hi_p, r, pr))
    sd = math.sqrt(v)
    return max(0.0, mu + sst.norm.ppf(lo_p) * sd), mu + sst.norm.ppf(hi_p) * sd


def chance_over(stat: str, mu: float, var: list[float], line: float) -> float:
    v = _v(mu, var)
    if stat in NB_STATS and v > mu > 0:
        r = mu * mu / (v - mu)
        return float(1 - sst.nbinom.cdf(math.floor(line), r, r / (r + mu)))
    return float(1 - sst.norm.cdf((math.floor(line) + 0.5 - mu) / math.sqrt(v)))


def coverage(stat: str, mu: np.ndarray, y: np.ndarray, var: list[float], q: float) -> float:
    """Share of results inside the q range; a result exactly on an edge of a
    whole-number range counts half (so discrete stats can be calibrated)."""
    lo_p, hi_p = (1 - q) / 2, 1 - (1 - q) / 2
    v = np.maximum((var[0] * mu + var[1] * mu * mu) * (var[2] if len(var) > 2 else 1.0), 1e-6)
    if stat in NB_STATS:
        nb = v > mu
        r = np.where(nb, mu * mu / np.maximum(v - mu, 1e-9), 1e6)
        pr = r / (r + np.maximum(mu, 1e-9))
        lo, hi = sst.nbinom.ppf(lo_p, r, pr), sst.nbinom.ppf(hi_p, r, pr)
        inside = (y > lo) & (y < hi)
        edge = (y == lo) | (y == hi)
        return float(inside.mean() + 0.5 * edge.mean())
    sd = np.sqrt(v)
    lo, hi = mu + sst.norm.ppf(lo_p) * sd, mu + sst.norm.ppf(hi_p) * sd
    return float(((y >= lo) & (y <= hi)).mean())


def calibrate_scale(stat: str, mu: np.ndarray, y: np.ndarray, var: list[float]) -> float:
    """Variance multiplier that makes the 80% range hold 80% of results."""
    best = (1.0, 9.0)
    for sc in np.arange(0.3, 1.61, 0.05):
        c = coverage(stat, mu, y, [var[0], var[1], sc], 0.8)
        if abs(c - 0.8) < best[1]:
            best = (float(sc), abs(c - 0.8))
    return round(best[0], 2)
