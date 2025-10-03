from __future__ import annotations

import csv
import json
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Ensure local src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from baseball_sim.features.context_features import context_features
from baseball_sim.features.ratings_adapter import batter_features, pitcher_features
from baseball_sim.features.transforms import z_0_99
from baseball_sim.models.pa_model import PaOutcomeModel
from baseball_sim.models.pitch_engine import PitchEngine
from baseball_sim.sampler.rng import sample_event
from baseball_sim.utils.parks import load_parks_yaml
import argparse
import httpx


@dataclass
class Player:
    player_id: str
    bats: str  # L/R/S
    primary_position: str
    other_positions: str
    is_pitcher: bool
    ratings: Dict[str, int] = field(default_factory=dict)


@dataclass
class Team:
    name: str
    lineup: List[str]  # batting order list of player_ids
    starters_by_pos: Dict[str, str]  # pos -> player_id
    bench: List[str]
    starter_pitcher: str
    bullpen: List[str]
    # derived/utility
    bats_map: Dict[str, str]
    ratings_map: Dict[str, Dict[str, int]]

    def defense_block(self) -> Dict[str, Dict[str, int] | Dict[str, str]]:
        # Build a minimal defense mapping: oaa_by_pos from fielder 'fa', arm_by_pos from 'arm' for C/LF/CF/RF
        oaa = {}
        arm = {}
        for pos, pid in self.starters_by_pos.items():
            r = self.ratings_map.get(pid, {})
            if r.get("fa") is not None:
                oaa[pos] = int(r.get("fa"))
            if pos in ("C", "LF", "CF", "RF") and (r.get("arm") is not None or r.get("as") is not None):
                arm[pos] = int(r.get("arm") if r.get("arm") is not None else r.get("as"))
        return {"oaa_by_pos": oaa, "arm_by_pos": arm, "alignment": {"infield": "normal", "outfield": "normal"}}


def read_players_csv(path: Path) -> Dict[str, Player]:
    players: Dict[str, Player] = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = (row.get("player_id") or "").strip()
            if not pid:
                continue
            bats = (row.get("bats") or "R").strip().upper()
            primary = (row.get("primary_position") or "").strip().upper()
            other = (row.get("other_positions") or "").strip().upper()
            is_pitcher = str(row.get("is_pitcher") or "0").strip() in {"1", "true", "True"}
            # Pull ratings columns present in utils.players._RATING_KEYS
            rating_keys = {
                "ch",
                "ph",
                "gf",
                "pl",
                "sc",
                "sp",
                "fa",
                "arm",
                "as",
                "endurance",
                "control",
                "movement",
                "hold_runner",
                "fb",
                "sl",
                "si",
                "cu",
                "cb",
                "scb",
                "kn",
            }
            ratings: Dict[str, int] = {}
            for k in rating_keys:
                if k in row and row[k] not in (None, ""):
                    try:
                        ratings[k] = int(float(row[k]))
                    except Exception:
                        pass
            players[pid] = Player(
                player_id=pid,
                bats=("L" if bats == "L" else "R"),  # normalize S->R
                primary_position=primary,
                other_positions=other,
                is_pitcher=is_pitcher,
                ratings=ratings,
            )
    return players


ALL_FIELD_POS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]


def can_play(player: Player, pos: str) -> bool:
    if player.primary_position == pos:
        return True
    if not player.other_positions:
        return False
    return pos in {p.strip().upper() for p in player.other_positions.split(",")}


def score_batter_for_order(p: Player) -> float:
    # Simple: blend contact and power
    ch = p.ratings.get("ch", 50)
    ph = p.ratings.get("ph", 50)
    spd = p.ratings.get("sp", 50)
    return 0.5 * ch + 0.4 * ph + 0.1 * spd


def score_pitcher_as_sp(p: Player) -> float:
    endu = p.ratings.get("endurance", 50)
    ctrl = p.ratings.get("control", 50)
    mov = p.ratings.get("movement", 50)
    arm = p.ratings.get("arm", p.ratings.get("as", 50))
    return 0.4 * endu + 0.3 * ctrl + 0.2 * mov + 0.1 * (arm or 50)


def score_pitcher_as_rp(p: Player) -> float:
    ctrl = p.ratings.get("control", 50)
    mov = p.ratings.get("movement", 50)
    arm = p.ratings.get("arm", p.ratings.get("as", 50))
    return 0.4 * ctrl + 0.4 * mov + 0.2 * (arm or 50)


def build_team(name: str, players: Dict[str, Player], seed: int = 1) -> Team:
    rng = random.Random(seed)
    hitters = [p for p in players.values() if not p.is_pitcher and p.primary_position != "P"]
    pitchers = [p for p in players.values() if p.is_pitcher or p.primary_position == "P"]

    # Assign starters by position
    starters_by_pos: Dict[str, str] = {}
    used: set[str] = set()
    for pos in ALL_FIELD_POS:
        eligible = [p for p in hitters if p.player_id not in used and can_play(p, pos)]
        if not eligible:
            # fallback: any hitter not used
            eligible = [p for p in hitters if p.player_id not in used]
        if not eligible:
            continue
        # Choose best by simple score
        eligible.sort(key=score_batter_for_order, reverse=True)
        choice = eligible[0]
        starters_by_pos[pos] = choice.player_id
        used.add(choice.player_id)

    # Fill remaining lineup slots (including DH) from best remaining hitters
    remaining_hitters = [p for p in hitters if p.player_id not in used]
    remaining_hitters.sort(key=score_batter_for_order, reverse=True)
    # Batting order: start with middle infield and CF near top if present
    order_candidates: List[Player] = []
    # Ensure we include the field starters in the batting order
    starter_objs = [players[pid] for pid in starters_by_pos.values() if pid in players]
    order_candidates.extend(starter_objs)
    # plus fill with best remaining
    order_candidates.extend(remaining_hitters)
    # Deduplicate while preserving order
    seen: set[str] = set()
    batting_order: List[str] = []
    for p in order_candidates:
        if p.player_id in seen:
            continue
        seen.add(p.player_id)
        batting_order.append(p.player_id)
        if len(batting_order) >= 9:
            break

    bench = [p.player_id for p in remaining_hitters if p.player_id not in seen][:5]

    # Select starter and bullpen
    pitchers_sorted_sp = sorted(pitchers, key=score_pitcher_as_sp, reverse=True)
    starter_pitcher = pitchers_sorted_sp[0].player_id if pitchers_sorted_sp else (pitchers[0].player_id if pitchers else "P_X")
    bullpen = [p.player_id for p in pitchers if p.player_id != starter_pitcher]
    # Order bullpen by RP score
    bullpen.sort(key=lambda pid: score_pitcher_as_rp(players[pid]), reverse=True)

    bats_map = {p.player_id: p.bats for p in players.values()}
    ratings_map = {p.player_id: p.ratings for p in players.values()}

    return Team(
        name=name,
        lineup=batting_order,
        starters_by_pos=starters_by_pos,
        bench=bench,
        starter_pitcher=starter_pitcher,
        bullpen=bullpen,
        bats_map=bats_map,
        ratings_map=ratings_map,
    )


def forced_advance_walk(bases: Dict[str, Optional[str]], batter_id: str) -> Tuple[Dict[str, Optional[str]], int]:
    # Simple forced-walk advancement; returns (new_bases, rbi)
    b = dict(bases)
    rbi = 0
    # If bases loaded, force in a run
    if b.get("1B") is not None and b.get("2B") is not None and b.get("3B") is not None:
        rbi = 1
    # Perform base advances back-to-front
    if b.get("3B") is not None and b.get("2B") is not None and b.get("1B") is not None:
        # everyone forced; runner from 3B scores (not represented in bases)
        b["3B"] = b["2B"]
        b["2B"] = b["1B"]
        b["1B"] = batter_id
        return b, rbi
    # Partial forces
    if b.get("2B") is not None and b.get("1B") is not None and b.get("3B") is None:
        b["3B"] = b["2B"]
        b["2B"] = b["1B"]
        b["1B"] = batter_id
        return b, rbi
    if b.get("1B") is not None and b.get("2B") is None:
        b["2B"] = b["1B"]
        b["1B"] = batter_id
        return b, rbi
    # Only 3B occupied or empty bases
    if b.get("1B") is None:
        b["1B"] = batter_id
    return b, rbi


def advance_on_hit(event: str, bases: Dict[str, Optional[str]], batter_id: str) -> Tuple[Dict[str, Optional[str]], int]:
    b = dict(bases)
    rbi = 0
    if event == "1B":
        # Conservative: force 3B scores, 2B->3B, 1B->2B
        if b.get("3B") is not None:
            rbi += 1
            b["3B"] = None
        if b.get("2B") is not None:
            b["3B"] = b["2B"]
            b["2B"] = None
        if b.get("1B") is not None:
            b["2B"] = b["1B"]
        b["1B"] = batter_id
    elif event == "2B":
        # Everyone advances two; 2B and 3B score; 1B -> 3B; batter -> 2B
        if b.get("3B") is not None:
            rbi += 1
        if b.get("2B") is not None:
            rbi += 1
        new_b3 = b.get("1B")  # runner from 1B goes to 3B if existed
        b["3B"] = new_b3
        b["2B"] = batter_id
        b["1B"] = None
    elif event == "3B":
        # All runners score; batter to 3B
        rbi += int(b.get("1B") is not None) + int(b.get("2B") is not None) + int(b.get("3B") is not None)
        b["1B"], b["2B"], b["3B"] = None, None, batter_id
    elif event == "HR":
        rbi += int(b.get("1B") is not None) + int(b.get("2B") is not None) + int(b.get("3B") is not None) + 1
        b["1B"], b["2B"], b["3B"] = None, None, None
    else:
        # Default: treat as single
        if b.get("3B") is not None:
            rbi += 1
            b["3B"] = None
        if b.get("2B") is not None:
            b["3B"] = b["2B"]
            b["2B"] = None
        if b.get("1B") is not None:
            b["2B"] = b["1B"]
        b["1B"] = batter_id
    return b, rbi


def build_features_for(batter: Dict, pitcher: Dict, state: Dict, park_feats: Dict) -> Dict[str, float]:
    feats = {}
    feats.update(batter_features(batter))
    feats.update(pitcher_features(pitcher))
    feats.update(context_features(state, pitcher, batter))
    feats.update(park_feats or {})
    # If batter speed exists, include bat_speed_z
    try:
        sp = None
        if isinstance(batter.get("ratings"), dict):
            sp = batter["ratings"].get("sp")
        if sp is not None:
            feats["bat_speed_z"] = z_0_99(sp)
    except Exception:
        pass
    # Gate RISP bonus by context
    if "bat_risp_z" in feats:
        try:
            risp_gate = float(feats.get("ctx_is_risp", 0) or 0.0)
        except Exception:
            risp_gate = 0.0
        feats["bat_risp_z"] = float(feats["bat_risp_z"] or 0.0) * risp_gate
    return feats


def _make_client(http_base: Optional[str]):
    if http_base:
        class HttpClient:
            def __init__(self, base: str):
                self.base = base.rstrip("/")
                self.sess = httpx.Client(timeout=30.0)
            def get(self, path: str):
                return self.sess.get(self.base + path)
            def post(self, path: str, json):
                return self.sess.post(self.base + path, json=json)
        return HttpClient(http_base)
    # In-process client
    from fastapi.testclient import TestClient
    from baseball_sim.serve.api import app
    return TestClient(app)


def simulate_game(game_id: str, home: Team, away: Team, park_id: str, seed: int = 1234, http_base: Optional[str] = None, strict: bool = True) -> Dict[str, object]:
    rng = random.Random(seed)
    np_seed = seed
    client = _make_client(http_base)

    # Health check -> enforce artifact availability when strict
    try:
        r = client.get("/health")
        body = r.json() if hasattr(r, "json") else r.json()
        model_available = bool(((body or {}).get("model") or {}).get("available", False))
        if strict and not model_available:
            raise RuntimeError("Model artifacts not loaded on server. Start with STRICT_MODE=1 and ensure training artifacts exist.")
    except Exception as e:
        if strict:
            raise

    # Game state
    inning = 1
    half = "T"  # T: away bats first
    outs = 0
    bases: Dict[str, Optional[str]] = {"1B": None, "2B": None, "3B": None}
    score = {"away": 0, "home": 0}
    # Batting order indices
    idx_home = 0
    idx_away = 0
    # Pitchers in use (simple: SP for 1-6, then two RP)
    cur_pitcher_home = home.starter_pitcher
    cur_pitcher_away = away.starter_pitcher
    bullpen_ptr_home = 0
    bullpen_ptr_away = 0

    # Logs
    pitch_log: List[Dict] = []
    game_events: List[Dict] = []  # for boxscore endpoint shape

    def current_team_and_pitcher() -> Tuple[Team, Team, str]:
        if half == "T":
            return away, home, cur_pitcher_home
        else:
            return home, away, cur_pitcher_away

    # Helper to switch pitchers by inning
    def maybe_switch_pitcher():
        nonlocal cur_pitcher_home, cur_pitcher_away, bullpen_ptr_home, bullpen_ptr_away
        if inning == 7 and half == "T" and home.bullpen:
            cur_pitcher_home = home.bullpen[min(bullpen_ptr_home, len(home.bullpen) - 1)]
            bullpen_ptr_home += 1
        if inning == 7 and half == "B" and away.bullpen:
            cur_pitcher_away = away.bullpen[min(bullpen_ptr_away, len(away.bullpen) - 1)]
            bullpen_ptr_away += 1
        if inning == 9 and half == "T" and home.bullpen and bullpen_ptr_home < len(home.bullpen):
            cur_pitcher_home = home.bullpen[min(bullpen_ptr_home, len(home.bullpen) - 1)]
            bullpen_ptr_home += 1
        if inning == 9 and half == "B" and away.bullpen and bullpen_ptr_away < len(away.bullpen):
            cur_pitcher_away = away.bullpen[min(bullpen_ptr_away, len(away.bullpen) - 1)]
            bullpen_ptr_away += 1

    # Play 9 innings plus extras as needed
    while True:
        # Half-inning loop
        outs = 0
        bases = {"1B": None, "2B": None, "3B": None}
        maybe_switch_pitcher()

        while outs < 3:
            batting_team, fielding_team, pit_id = current_team_and_pitcher()
            # Select batter
            if half == "T":
                bat_pid = batting_team.lineup[idx_away % len(batting_team.lineup)]
                idx_away += 1
            else:
                bat_pid = batting_team.lineup[idx_home % len(batting_team.lineup)]
                idx_home += 1

            batter = {
                "player_id": bat_pid,
                "bats": batting_team.bats_map.get(bat_pid, "R"),
                "ratings": batting_team.ratings_map.get(bat_pid, {}),
            }
            pitcher = {
                "player_id": pit_id,
                "throws": fielding_team.bats_map.get(pit_id, "R"),  # throws not in CSV; approximate with bats
                "ratings": fielding_team.ratings_map.get(pit_id, {}),
            }
            state = {
                "inning": inning,
                "half": half,
                "outs": outs,
                "bases": bases,
                "score": score,
                "count": {"balls": 0, "strikes": 0},
                "park_id": park_id,
                "rules": {"dh": True, "shift_restrictions": True, "three_batter_min": True},
            }
            # Construct defense block for the fielding team
            defense = fielding_team.defense_block()
            # Call API to sample the PA outcome using trained model + calibrator
            req = {
                "game_id": game_id,
                "state": {
                    "inning": state["inning"],
                    "half": state["half"],
                    "outs": state["outs"],
                    "bases": state["bases"],
                    "score": state["score"],
                    "count": state["count"],
                    "park_id": park_id,
                    "rules": {"dh": True, "shift_restrictions": True, "three_batter_min": True},
                },
                "batter": {"player_id": batter["player_id"], "bats": batter["bats"], "ratings": batter.get("ratings", {})},
                "pitcher": {"player_id": pitcher["player_id"], "throws": pitcher["throws"], "ratings": pitcher.get("ratings", {})},
                "defense": defense,
                "return_probabilities": True,
                "return_contact": True,
                "seed": np_seed,
            }
            r = client.post("/v1/sim/plate-appearance", json=req)
            body = r.json()
            event = body.get("sampled_event")
            np_seed += 1

            # Build a plausible pitch-by-pitch sequence that ends with event
            cb, cs = 0, 0
            pitch_seq: List[Tuple[str, str, int, int]] = []  # (pitch_type, result, b, s)
            # Very simple pitch type from engine (deterministic argmax)
            pitch_type = PitchEngine._select_pitch_type(pitcher.get("ratings", {}))

            def add_pitch(res: str):
                nonlocal cb, cs
                pitch_seq.append((pitch_type, res, cb, cs))
                if res == "ball":
                    cb += 1
                elif res in ("called_strike", "swing_miss"):
                    cs += 1
                elif res == "foul":
                    cs = min(cs + 1, 2)

            # Random preamble of up to 3 pitches
            pre_pitches = random.randint(0, 3)
            for _ in range(pre_pitches):
                res = random.choices(["ball", "called_strike", "foul", "swing_miss"], weights=[4, 3, 2, 2])[0]
                # avoid finishing AB prematurely
                if event in {"BB", "HBP"} and cb >= 3:
                    break
                if event == "K" and cs >= 2:
                    break
                add_pitch(res)

            # Terminal pitch based on event
            if event == "BB":
                while cb < 4:
                    add_pitch("ball")
                # Apply walk
                new_bases, rbi = forced_advance_walk(bases, bat_pid)
                bases = new_bases
                score_key = "away" if half == "T" else "home"
                score[score_key] += rbi
                # Log game event for boxscore
                game_events.append({
                    "inning": inning,
                    "half": half,
                    "team": "away" if half == "T" else "home",
                    "rbi": rbi,
                    "outs_recorded": 0,
                    "error": False,
                })
            elif event == "HBP":
                add_pitch("hbp")
                new_bases, rbi = forced_advance_walk(bases, bat_pid)
                bases = new_bases
                score_key = "away" if half == "T" else "home"
                score[score_key] += rbi
                game_events.append({
                    "inning": inning,
                    "half": half,
                    "team": "away" if half == "T" else "home",
                    "rbi": rbi,
                    "outs_recorded": 0,
                    "error": False,
                })
            elif event == "K":
                while cs < 3:
                    add_pitch(random.choice(["called_strike", "swing_miss"]))
                outs += 1
                game_events.append({
                    "inning": inning,
                    "half": half,
                    "team": "away" if half == "T" else "home",
                    "rbi": 0,
                    "outs_recorded": 1,
                    "error": False,
                })
            elif event in {"IP_OUT"}:
                # Put ball in play and record an out
                add_pitch("in_play")
                outs += 1
                game_events.append({
                    "inning": inning,
                    "half": half,
                    "team": "away" if half == "T" else "home",
                    "rbi": 0,
                    "outs_recorded": 1,
                    "error": False,
                })
            elif event in {"1B", "2B", "3B", "HR"}:
                add_pitch("in_play")
                new_bases, rbi = advance_on_hit(event, bases, bat_pid)
                bases = new_bases
                score_key = "away" if half == "T" else "home"
                score[score_key] += rbi
                # Outs unchanged on hit
                game_events.append({
                    "inning": inning,
                    "half": half,
                    "team": "away" if half == "T" else "home",
                    "rbi": rbi,
                    "outs_recorded": 0,
                    "error": False,
                })
            else:
                # Fallback: treat as in-play out
                add_pitch("in_play")
                outs += 1
                game_events.append({
                    "inning": inning,
                    "half": half,
                    "team": "away" if half == "T" else "home",
                    "rbi": 0,
                    "outs_recorded": 1,
                    "error": False,
                })

            # Append pitch log rows
            for (ptype, res, b0, s0) in pitch_seq:
                pitch_log.append({
                    "game_id": game_id,
                    "inning": inning,
                    "half": half,
                    "batter_id": bat_pid,
                    "pitcher_id": pit_id,
                    "count_b": b0,
                    "count_s": s0,
                    "pitch_type": ptype,
                    "result": res,
                    "park_id": park_id,
                })

        # Switch halves or innings
        if half == "T":
            half = "B"
        else:
            # End of full inning
            if inning >= 9 and score["home"] != score["away"]:
                # Game ends after 9 if not tied
                break
            inning += 1
            half = "T"

        # End condition after extras: if completed bottom of an extra inning and scores differ, stop
        if inning > 9 and half == "T" and score["home"] != score["away"]:
            break

    return {
        "pitch_log": pitch_log,
        "game_events": game_events,
        "final_score": score,
        "park_id": park_id,
    }


def choose_random_park_id() -> str:
    cfg = load_parks_yaml(ROOT / "config" / "parks.yaml")
    if cfg:
        return random.choice(list(cfg.keys()))
    # Fallback to raw seamheads CSV if PyYAML not available
    try:
        path = ROOT / "data" / "raw" / "seamheads_ballparks.csv"
        if path.exists():
            with path.open(newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                ids = [row["retro_park_id"] for row in r if row.get("retro_park_id")]
                if ids:
                    return random.choice(ids)
    except Exception:
        pass
    return "GEN"


def main():
    ap = argparse.ArgumentParser(description="Simulate a full game from players.csv via API (trained model required)")
    ap.add_argument("--http", help="Base URL (e.g., http://localhost:8000). If omitted, runs in-process TestClient.")
    ap.add_argument("--strict", action="store_true", default=True, help="Fail if server lacks model artifacts.")
    ap.add_argument("--game-id", default="SIM-CSV-GAME-1")
    args = ap.parse_args()

    # If running in-process strict mode, propagate env so API raises if missing
    if not args.http and args.strict:
        os.environ["STRICT_MODE"] = "1"

    random.seed(42)
    game_id = args.game_id
    players = read_players_csv(ROOT / "data" / "players.csv")

    # Split players roughly evenly into two pools for sample teams
    all_ids = list(players.keys())
    random.shuffle(all_ids)
    mid = len(all_ids) // 2
    pool_away = {pid: players[pid] for pid in all_ids[:mid]}
    pool_home = {pid: players[pid] for pid in all_ids[mid:]}

    away = build_team("AWAY", pool_away, seed=1)
    home = build_team("HOME", pool_home, seed=2)

    park_id = choose_random_park_id()
    result = simulate_game(game_id, home, away, park_id, seed=123, http_base=args.http, strict=bool(args.strict))

    outdir = ROOT / "outputs"
    outdir.mkdir(parents=True, exist_ok=True)

    # Write pitch-by-pitch log CSV
    pitch_path = outdir / f"{game_id}_pitch_log.csv"
    with pitch_path.open("w", newline="", encoding="utf-8") as f:
        cols = [
            "game_id",
            "inning",
            "half",
            "batter_id",
            "pitcher_id",
            "count_b",
            "count_s",
            "pitch_type",
            "result",
            "park_id",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in result["pitch_log"]:
            w.writerow({k: row.get(k) for k in cols})

    # Write a simple box score text and JSON
    score = result["final_score"]
    box_txt = outdir / f"{game_id}_boxscore.txt"
    with box_txt.open("w", encoding="utf-8") as f:
        f.write(f"Game ID: {game_id}\n")
        f.write(f"Ballpark: {result['park_id']}\n")
        f.write(f"Away ({away.name}): {score['away']}\n")
        f.write(f"Home ({home.name}): {score['home']}\n")
        f.write("\n")
        f.write("Away Lineup (batting order):\n")
        for i, pid in enumerate(away.lineup, 1):
            f.write(f"  {i}. {pid}\n")
        f.write("Away Starters by Position:\n")
        for pos in ALL_FIELD_POS:
            pid = away.starters_by_pos.get(pos, "-")
            f.write(f"  {pos}: {pid}\n")
        f.write(f"Away Starter: {away.starter_pitcher}\n")
        f.write(f"Away Bullpen: {', '.join(away.bullpen)}\n")
        f.write(f"Away Bench: {', '.join(away.bench)}\n")
        f.write("\n")
        f.write("Home Lineup (batting order):\n")
        for i, pid in enumerate(home.lineup, 1):
            f.write(f"  {i}. {pid}\n")
        f.write("Home Starters by Position:\n")
        for pos in ALL_FIELD_POS:
            pid = home.starters_by_pos.get(pos, "-")
            f.write(f"  {pos}: {pid}\n")
        f.write(f"Home Starter: {home.starter_pitcher}\n")
        f.write(f"Home Bullpen: {', '.join(home.bullpen)}\n")
        f.write(f"Home Bench: {', '.join(home.bench)}\n")
        f.write("\nFinal: Away {} - Home {}\n".format(score['away'], score['home']))

    # Also dump a machine-readable summary
    with (outdir / f"{game_id}_summary.json").open("w", encoding="utf-8") as f:
        json.dump({
            "game_id": game_id,
            "park_id": result["park_id"],
            "teams": {
                "away": {
                    "name": away.name,
                    "lineup": away.lineup,
                    "starters_by_pos": away.starters_by_pos,
                    "bench": away.bench,
                    "starter_pitcher": away.starter_pitcher,
                    "bullpen": away.bullpen,
                },
                "home": {
                    "name": home.name,
                    "lineup": home.lineup,
                    "starters_by_pos": home.starters_by_pos,
                    "bench": home.bench,
                    "starter_pitcher": home.starter_pitcher,
                    "bullpen": home.bullpen,
                },
            },
            "final_score": score,
        }, f, indent=2)

    print("Wrote:")
    print(" -", pitch_path)
    print(" -", box_txt)
    print(" -", (outdir / f"{game_id}_summary.json"))


if __name__ == "__main__":
    raise SystemExit(main())
