"""
Live sources, all public and free (no keys). Same endpoints Prop Edge has
used since Oct 2026.

  injuries  ESPN injury report      who is Out / Doubtful / Questionable
  rosters   ESPN team rosters       who is on which team today (trades)
  kalshi    Kalshi KXNBAGAME        the market's win chance for each game

Every fetch fails soft: on any error the last good copy in data/ is kept,
and the error is recorded so the analyst can say the data is stale.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from teleios import data, teams

UA = {"User-Agent": "Mozilla/5.0 (NBA Teleios Analyst)"}
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"

INJ_PATH = data.DATA / "injuries.json"
ROSTER_PATH = data.DATA / "rosters.json"
KALSHI_PATH = data.DATA / "kalshi_games.json"

# out: removed from the lineup. questionable: kept, but flagged.
STATUS_CLASS = {
    "out": "out", "doubtful": "out", "suspension": "out", "suspended": "out",
    "questionable": "questionable", "day-to-day": "questionable", "game time decision": "questionable",
    "probable": "probable",
}


def _load(path: Path, empty: dict) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return empty


def _keep_old(path: Path, empty: dict, err: Exception) -> dict:
    prev = _load(path, empty)
    prev["error"] = f"{type(err).__name__}: {str(err)[:160]} (using copy from {prev.get('fetched_at') or 'never'})"
    return prev


# --- injuries ---------------------------------------------------------------

def fetch_injuries() -> dict:
    empty = {"fetched_at": None, "players": []}
    try:
        r = requests.get(f"{ESPN}/injuries", timeout=20, headers=UA)
        r.raise_for_status()
        players = []
        for block in r.json().get("injuries") or []:
            for inj in block.get("injuries") or []:
                ath = inj.get("athlete") or {}
                pid = ath.get("id")
                if not pid:
                    for link in ath.get("links") or []:
                        m = re.search(r"/id/(\d+)", link.get("href") or "")
                        if m:
                            pid = m.group(1)
                            break
                status = str(inj.get("status") or "").strip()
                players.append({
                    "player": ath.get("displayName"),
                    "pid": int(pid) if pid else None,
                    "team": teams.canon((ath.get("team") or {}).get("abbreviation"), "espn"),
                    "status": status,
                    "class": STATUS_CLASS.get(status.lower(), "questionable" if status else None),
                    "comment": (inj.get("shortComment") or "")[:240],
                    "date": inj.get("date"),
                })
        if not players:
            raise ValueError("injury report came back empty")
        out = {"fetched_at": data.now_utc(), "error": None, "players": players}
        INJ_PATH.write_text(json.dumps(out, indent=1))
        return out
    except Exception as e:  # noqa: BLE001
        return _keep_old(INJ_PATH, empty, e)


def load_injuries() -> dict:
    return _load(INJ_PATH, {"fetched_at": None, "players": []})


# --- rosters ----------------------------------------------------------------

def fetch_rosters() -> dict:
    empty = {"fetched_at": None, "players": []}
    try:
        r = requests.get(f"{ESPN}/teams", timeout=20, headers=UA)
        r.raise_for_status()
        tlist = [((t.get("team") or {}).get("id"), (t.get("team") or {}).get("abbreviation"))
                 for sp in r.json().get("sports", []) for lg in sp.get("leagues", []) for t in lg.get("teams", [])]
        players, failed = [], []
        for tid, abbr in tlist:
            code = teams.canon(abbr, "espn")
            if not tid or not code:
                continue
            try:
                rr = requests.get(f"{ESPN}/teams/{tid}/roster", timeout=20, headers=UA)
                rr.raise_for_status()
                for a in rr.json().get("athletes", []):
                    for it in (a.get("items") if isinstance(a, dict) and "items" in a else [a]):
                        players.append({"player": it.get("displayName"), "pid": int(it.get("id") or 0),
                                        "team": code, "pos": (it.get("position") or {}).get("abbreviation")})
            except Exception:  # noqa: BLE001
                failed.append(code)
            time.sleep(0.15)
        if len(players) < 300:
            raise ValueError(f"only {len(players)} players returned")
        out = {"fetched_at": data.now_utc(), "error": None, "failed_teams": failed, "players": players}
        ROSTER_PATH.write_text(json.dumps(out, indent=1))
        return out
    except Exception as e:  # noqa: BLE001
        return _keep_old(ROSTER_PATH, empty, e)


def load_rosters() -> dict:
    return _load(ROSTER_PATH, {"fetched_at": None, "players": []})


# --- Kalshi game markets ----------------------------------------------------

MAX_GAP = 0.10  # widest bid/ask gap still counted as a real price


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _prices(m: dict):
    bid, ask = _num(m.get("yes_bid_dollars")), _num(m.get("yes_ask_dollars"))
    if bid is None and m.get("yes_bid") is not None:
        bid = _num(m["yes_bid"]) / 100
    if ask is None and m.get("yes_ask") is not None:
        ask = _num(m["yes_ask"]) / 100
    bid = bid if bid and bid > 0 else None
    ask = ask if ask and ask < 1 else None
    mid = (bid + ask) / 2 if bid and ask and ask - bid <= MAX_GAP else None
    return bid, ask, mid


def parse_event(ticker: str):
    """'KXNBAGAME-26OCT20BOSDET' -> ('BOS', 'DET', '2026-10-20')."""
    m = re.search(r"-(\d{2})([A-Z]{3})(\d{2})([A-Z]+)$", ticker or "")
    if not m:
        return None
    yy, mon, dd, codes = m.groups()
    months = "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split()
    if mon not in months:
        return None
    day = f"20{yy}-{months.index(mon) + 1:02d}-{dd}"
    for split in (3, 2, 4):
        a, h = teams.canon(codes[:split], "kalshi"), teams.canon(codes[split:], "kalshi")
        if a and h:
            return a, h, day
    return None


def _get(path, params):
    for attempt in range(4):
        r = requests.get(f"{KALSHI}{path}", params=params, timeout=20, headers=UA)
        if r.status_code == 429 and attempt < 3:
            time.sleep(1.5 * 2 ** attempt)
            continue
        r.raise_for_status()
        return r.json()


def fetch_kalshi() -> dict:
    """Open KXNBAGAME events -> one record per game with the market's
    chance for the home team (midpoint of a tight quote, normalised so both
    sides add to 100%)."""
    empty = {"fetched_at": None, "games": []}
    try:
        events, cursor = [], None
        for _ in range(10):
            p = {"series_ticker": "KXNBAGAME", "status": "open", "limit": 200, "with_nested_markets": "true"}
            if cursor:
                p["cursor"] = cursor
            d = _get("/events", p)
            events += d.get("events") or []
            cursor = d.get("cursor")
            if not cursor or not d.get("events"):
                break
        games = []
        for ev in events:
            t = parse_event(ev.get("event_ticker"))
            if not t:
                continue
            away, home, day = t
            mk = ev.get("markets")
            if mk is None:
                mk = (_get("/markets", {"event_ticker": ev["event_ticker"], "limit": 10}) or {}).get("markets") or []
            side = {}
            for m in mk:
                suffix = (m.get("ticker") or "").rsplit("-", 1)[-1]
                code = teams.canon(suffix, "kalshi")
                if code in (home, away):
                    side[code] = _prices(m)
            mh = side.get(home, (None, None, None))[2]
            ma = side.get(away, (None, None, None))[2]
            if mh is None and ma is None:
                p_home = None
            elif mh is not None and ma is not None:
                p_home = mh / (mh + ma)
            else:
                p_home = mh if mh is not None else 1 - ma
            games.append({"date": day, "home": home, "away": away, "event": ev.get("event_ticker"),
                          "p_home": None if p_home is None else round(p_home, 4),
                          "home_bid_ask": side.get(home, (None, None))[:2],
                          "away_bid_ask": side.get(away, (None, None))[:2]})
        out = {"fetched_at": data.now_utc(), "error": None, "games": games}
        KALSHI_PATH.write_text(json.dumps(out, indent=1))
        return out
    except Exception as e:  # noqa: BLE001
        return _keep_old(KALSHI_PATH, empty, e)


# --- point spreads ----------------------------------------------------------
# Kalshi KXNBASPREAD: one market per "TEAM wins by over X.5". The main line is
# the tightly quoted market priced nearest 50%. ESPN's scoreboard odds
# ("MEM -3.5") are the fallback when Kalshi has no usable quote.
SPREADS_PATH = data.DATA / "spreads.json"


def _espn_spreads(days: list[str]) -> dict:
    out = {}
    for day in days:
        try:
            r = requests.get(f"{ESPN}/scoreboard", params={"dates": day.replace("-", "")}, timeout=20, headers=UA)
            r.raise_for_status()
            for ev in r.json().get("events", []):
                comp = (ev.get("competitions") or [{}])[0]
                side = {c.get("homeAway"): teams.canon((c.get("team") or {}).get("abbreviation"), "espn")
                        for c in comp.get("competitors", [])}
                home, away = side.get("home"), side.get("away")
                odds = (comp.get("odds") or [{}])[0]
                det = (odds.get("details") or "").strip()
                if not home or not away or not det:
                    continue
                if det.upper().startswith("EVEN"):
                    line = 0.0
                else:
                    m = re.match(r"^([A-Za-z]+)\s+([-+]?\d+(?:\.\d+)?)$", det)
                    if not m:
                        continue
                    fav = teams.canon(m.group(1), "espn")
                    v = float(m.group(2))
                    if fav == home:
                        line = v
                    elif fav == away:
                        line = -v
                    else:
                        continue
                out[(day, home, away)] = {"date": day, "home": home, "away": away, "home_line": line,
                                          "market_p_home_cover": None, "source": f"ESPN ({odds.get('provider', {}).get('name') or 'consensus'})"}
        except Exception:  # noqa: BLE001
            continue
    return out


def fetch_spreads(days: list[str]) -> dict:
    """Main point spread for each game, from the home team's side
    (home_line -4.5 = home favoured by 4.5), plus the market's chance the
    home team covers when Kalshi quotes it."""
    empty = {"fetched_at": None, "games": []}
    try:
        games, diag = {}, {}
        try:
            events, cursor = [], None
            for _ in range(10):
                p = {"series_ticker": "KXNBASPREAD", "status": "open", "limit": 200, "with_nested_markets": "true"}
                if cursor:
                    p["cursor"] = cursor
                d = _get("/events", p)
                events += d.get("events") or []
                cursor = d.get("cursor")
                if not cursor or not d.get("events"):
                    break
            for ev in events:
                t = parse_event(ev.get("event_ticker"))
                if not t:
                    continue
                away, home, day = t
                mk = ev.get("markets")
                if mk is None:
                    mk = (_get("/markets", {"event_ticker": ev["event_ticker"], "limit": 200}) or {}).get("markets") or []
                cands = []
                for m in mk:
                    tail = (m.get("ticker") or "").rsplit("-", 1)[-1]
                    mm = re.match(r"^([A-Z]+?)(\d+)$", tail)
                    if not mm:
                        continue
                    team = teams.canon(mm.group(1), "kalshi")
                    strike = _num(m.get("floor_strike"))
                    if strike is None:
                        strike = int(mm.group(2)) - 0.5
                    _, _, mid = _prices(m)
                    if team in (home, away) and mid is not None and 0.2 <= mid <= 0.8:
                        cands.append((abs(mid - 0.5), team, float(strike), mid))
                if not cands:
                    continue
                _, team, strike, mid = min(cands)
                if team == home:
                    line, pc = -strike, mid
                else:
                    line, pc = strike, 1 - mid
                games[(day, home, away)] = {"date": day, "home": home, "away": away, "home_line": line,
                                            "market_p_home_cover": round(pc, 4), "source": "Kalshi", "event": ev.get("event_ticker")}
            diag["kalshi"] = f"{len(events)} events, {len(games)} main lines"
        except Exception as e:  # noqa: BLE001
            diag["kalshi"] = f"failed: {type(e).__name__}: {str(e)[:120]}"
        espn = _espn_spreads(days)
        diag["espn"] = f"{len(espn)} lines"
        for k, v in espn.items():
            games.setdefault(k, v)
        out = {"fetched_at": data.now_utc(), "error": None, "diagnostics": diag, "games": list(games.values())}
        SPREADS_PATH.write_text(json.dumps(out, indent=1))
        return out
    except Exception as e:  # noqa: BLE001
        return _keep_old(SPREADS_PATH, empty, e)


def load_spreads() -> dict:
    return _load(SPREADS_PATH, {"fetched_at": None, "games": []})


def load_kalshi() -> dict:
    return _load(KALSHI_PATH, {"fetched_at": None, "games": []})


# --- Kalshi player prop ladders ---------------------------------------------
# Each rung "N+" (e.g. "Ty Jerome: 8+ assists") is the same bet as OVER N-0.5.
# The ladder's main line is the tightly quoted rung priced nearest 50%, and
# only rungs whose OVER chance is between 25% and 80% count (Prop Edge's rule).

PROP_SERIES = {"KXNBAPTS": "pts", "KXNBAREB": "reb", "KXNBAAST": "ast", "KXNBA3PT": "fg3m", "KXNBAPRA": "pra"}
PROPS_PATH = data.DATA / "kalshi_props.json"
_RUNG_RE = re.compile(r"^(?P<name>.+?)(?::|\s+-)?\s+(?P<n>\d+)\+")


def norm_name(s: str | None) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    s = re.sub(r"[.'`]", "", s)
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"[^a-z]+", " ", s).strip()


def _rung(m: dict):
    n = _num(m.get("floor_strike"))
    n = None if n is None else (int(n) if float(n).is_integer() else int(n + 0.5))
    tail = (m.get("ticker") or "").rsplit("-", 1)[-1]
    if n is None and tail.isdigit():
        n = int(tail)
    for f in ("yes_sub_title", "title", "subtitle"):
        mm = _RUNG_RE.match((m.get(f) or "").strip())
        if mm:
            name = re.sub(r"^will\s+", "", mm.group("name").strip(), flags=re.I)
            return name, (n if n is not None else int(mm.group("n")))
    return None, n


def fetch_kalshi_props() -> dict:
    empty = {"fetched_at": None, "ladders": []}
    try:
        ladders: dict = {}
        diag = {}
        for series, stat in PROP_SERIES.items():
            events, cursor = [], None
            for _ in range(10):
                p = {"series_ticker": series, "status": "open", "limit": 200, "with_nested_markets": "true"}
                if cursor:
                    p["cursor"] = cursor
                d = _get("/events", p)
                events += d.get("events") or []
                cursor = d.get("cursor")
                if not cursor or not d.get("events"):
                    break
            n_m = 0
            for ev in events:
                t = parse_event(ev.get("event_ticker"))
                if not t:
                    continue
                away, home, day = t
                mk = ev.get("markets")
                if mk is None:
                    mk = (_get("/markets", {"event_ticker": ev["event_ticker"], "status": "open", "limit": 1000}) or {}).get("markets") or []
                    time.sleep(0.25)
                for m in mk:
                    name, n = _rung(m)
                    if not name or n is None:
                        continue
                    bid, ask, mid = _prices(m)
                    key = (norm_name(name), stat, day, home, away)
                    lad = ladders.setdefault(key, {"player": name, "stat": stat, "date": day, "home": home, "away": away, "rungs": []})
                    lad["rungs"].append({"n": n, "line": n - 0.5, "bid": bid, "ask": ask, "mid": mid})
                    n_m += 1
            diag[series] = f"{len(events)} events, {n_m} rungs"
            time.sleep(0.3)
        out_l = []
        for lad in ladders.values():
            lad["rungs"].sort(key=lambda r: r["n"])
            ok = [r for r in lad["rungs"] if r["mid"] is not None and 0.25 <= r["mid"] <= 0.80]
            lad["main"] = min(ok, key=lambda r: abs(r["mid"] - 0.5)) if ok else None
            out_l.append(lad)
        out = {"fetched_at": data.now_utc(), "error": None, "diagnostics": diag, "ladders": out_l}
        PROPS_PATH.write_text(json.dumps(out, indent=0))
        return out
    except Exception as e:  # noqa: BLE001
        return _keep_old(PROPS_PATH, empty, e)


def load_kalshi_props() -> dict:
    return _load(PROPS_PATH, {"fetched_at": None, "ladders": []})
