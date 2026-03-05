"""
FPL Gameweek Score Predictor
=============================
Data sources (all live from FPL API):
  bootstrap      : /api/bootstrap-static/
  fixtures       : /api/fixtures/
  player history : /api/element-summary/{id}/  ← exact per-GW points + minutes

Key improvements over previous version:
  1. Exact FPL points from element-summary API (no reconstruction from fixture stats)
  2. Exact minutes per GW (fixes availability score accuracy)
  3. Separate XGBoost model per position (GK / DEF / MID / FWD)
  4. Position-specific features (e.g. clean sheets for GK/DEF, goals for FWD)
  5. Best players ranked per position in output

Usage:
    python fpl_predictor.py [--predict-gw 31] [--debug "Salah"]

Output:
    data/fpl_predictions_gw<N>.csv        — all players ranked by PredPts
    data/fpl_best_by_position_gw<N>.csv   — top 10 per position
    data/fpl_validation_summary_gw<N>.csv — per-player MAE on val GWs
    data/fpl_training_summary.txt         — split info and per-position MAE
"""

import argparse
import io
import json
import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings("ignore")
os.makedirs("data", exist_ok=True)
os.makedirs(os.path.join("data", "predictions"), exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--predict-gw",    type=int, default=None)
parser.add_argument("--debug",         default="", help="Player web_name for detailed trace")
parser.add_argument("--force-refresh", action="store_true",
                    help="Ignore cache and re-fetch all data from FPL API")
parser.add_argument("--export",    action="store_true",
                    help="Upload predictions + cache to S3 after running")
parser.add_argument("--s3-bucket", default="",
                    help="S3 bucket name for --export  e.g. my-fpl-bucket")
args = parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# S3 upload helpers  (only used when --export is passed)
# ─────────────────────────────────────────────────────────────────────────────
def s3_client():
    try:
        import boto3
        return boto3.client("s3")
    except ImportError:
        raise SystemExit("❌ boto3 not installed. Run: pip install boto3")

def upload_file_to_s3(local_path, s3_key, bucket):
    s3 = s3_client()
    s3.upload_file(local_path, bucket, s3_key)
    print(f"  ☁️  s3://{bucket}/{s3_key}")

def upload_df_to_s3(df, s3_key, bucket, fmt="csv"):
    s3 = s3_client()
    buf = io.BytesIO()
    if fmt == "parquet":
        df.to_parquet(buf, index=False)
    else:
        df.to_csv(buf, index=False)
    buf.seek(0)
    s3.put_object(Bucket=bucket, Key=s3_key, Body=buf.getvalue())
    print(f"  ☁️  s3://{bucket}/{s3_key}")

def upload_json_to_s3(obj, s3_key, bucket):
    s3 = s3_client()
    s3.put_object(
        Bucket=bucket, Key=s3_key,
        Body=json.dumps(obj).encode()
    )
    print(f"  ☁️  s3://{bucket}/{s3_key}")

# ─────────────────────────────────────────────────────────────────────────────
# Cache paths  (local — swap read/write calls for boto3 S3 calls on AWS)
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR       = os.path.join("data", "cache")
CACHE_META      = os.path.join(CACHE_DIR, "meta.json")           # last cached GW
CACHE_BOOTSTRAP = os.path.join(CACHE_DIR, "bootstrap.json")
CACHE_FIXTURES  = os.path.join(CACHE_DIR, "fixtures.json")
CACHE_HISTORY   = os.path.join(CACHE_DIR, "player_history.parquet")
os.makedirs(CACHE_DIR, exist_ok=True)


def _live_last_gw():
    bs = requests.get(
        "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=15
    ).json()
    finished = [e["id"] for e in bs.get("events", []) if e.get("finished")]
    return max(finished) if finished else 0, bs


def cache_valid():
    if not all(os.path.exists(p) for p in
               [CACHE_META, CACHE_BOOTSTRAP, CACHE_FIXTURES, CACHE_HISTORY]):
        return False, None
    try:
        live_gw, live_bs = _live_last_gw()
        with open(CACHE_META) as f:
            cached_gw = json.load(f).get("last_finished_gw", -1)
        if live_gw == cached_gw:
            print(f"  Cache hit  — last finished GW = {live_gw}, no re-fetch needed")
            return True, live_bs          # still return live_bs (tiny, already fetched)
        print(f"  Cache miss — cached GW {cached_gw} -> live GW {live_gw}, re-fetching")
        return False, live_bs
    except Exception as ex:
        print(f"  Cache check failed ({ex}), re-fetching")
        return False, None


def save_cache(bs, fx, hist_df, gw):
    with open(CACHE_BOOTSTRAP, "w") as f:
        json.dump(bs, f)
    with open(CACHE_FIXTURES, "w") as f:
        json.dump(fx, f)
    hist_df.to_parquet(CACHE_HISTORY, index=False)
    with open(CACHE_META, "w") as f:
        json.dump({"last_finished_gw": int(gw)}, f)  # cast to int — numpy int32 is not JSON serializable
    print(f"  Cache saved for GW {gw}")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load bootstrap + fixtures  (from cache or FPL API)
# ─────────────────────────────────────────────────────────────────────────────
print("Checking cache...")
is_cached, prefetched_bs = (False, None) if args.force_refresh else cache_valid()

if is_cached:
    print("Loading bootstrap and fixtures from cache...")
    with open(CACHE_BOOTSTRAP) as f:
        bootstrap = json.load(f)
    with open(CACHE_FIXTURES) as f:
        fixtures_raw = json.load(f)
else:
    print("Fetching bootstrap and fixtures from FPL API...")
    try:
        bootstrap    = prefetched_bs or requests.get(
            "https://fantasy.premierleague.com/api/bootstrap-static/", timeout=15).json()
        fixtures_raw = requests.get(
            "https://fantasy.premierleague.com/api/fixtures/", timeout=15).json()
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"  ❌ API error: {e}")

players  = pd.DataFrame(bootstrap["elements"])
teams_df = pd.DataFrame(bootstrap["teams"])
fixtures = pd.DataFrame(fixtures_raw)

players["id"] = pd.to_numeric(players["id"], errors="coerce")
fixtures["event"] = pd.to_numeric(fixtures["event"], errors="coerce")

print(f"  Players : {len(players)}  |  Fixtures : {len(fixtures)}  |  Teams : {len(teams_df)}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Identify GWs and split
# ─────────────────────────────────────────────────────────────────────────────
finished_mask       = fixtures["finished"].astype(str).str.lower() == "true"
finished_gws_sorted = sorted(fixtures[finished_mask]["event"].dropna().unique().astype(int))
last_finished_gw    = finished_gws_sorted[-1] if finished_gws_sorted else 0
predict_gw          = args.predict_gw if args.predict_gw else last_finished_gw + 1

n_gws       = len(finished_gws_sorted)
n_train_gws = max(1, int(round(n_gws * 0.80)))
train_gws   = finished_gws_sorted[:n_train_gws]
val_gws     = finished_gws_sorted[n_train_gws:]

print(f"\n  Finished GWs    : GW{finished_gws_sorted[0]}–GW{last_finished_gw}  ({n_gws} total)")
print(f"  Train  (80%)    : GW{train_gws[0]}–GW{train_gws[-1]}  ({len(train_gws)} GWs)")
print(f"  Val    (20%)    : " + (f"GW{val_gws[0]}–GW{val_gws[-1]}  ({len(val_gws)} GWs)" if val_gws else "none yet"))
print(f"  Predicting      : GW{predict_gw}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Load exact per-GW history  (from cache or element-summary API)
#
#    element-summary gives us exact: total_points, minutes, goals_scored,
#    assists, clean_sheets, bonus, saves etc. — calculated by FPL directly.
#    On a cache hit this step takes <1 second instead of ~2 minutes.
# ─────────────────────────────────────────────────────────────────────────────
all_player_ids = players["id"].dropna().astype(int).tolist()

if is_cached:
    history = pd.read_parquet(CACHE_HISTORY)
    print(f"  Loaded {len(history)} player-GW records from cache")
else:
    print(f"\nFetching per-GW history for {len(players)} players (~2 min)...")
    all_histories = []

    for i, pid in enumerate(all_player_ids):
        try:
            url  = f"https://fantasy.premierleague.com/api/element-summary/{pid}/"
            data = requests.get(url, timeout=10).json()
            hist = pd.DataFrame(data.get("history", []))
            if not hist.empty:
                hist["player_id"] = pid
                all_histories.append(hist)
        except Exception:
            pass
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_player_ids)} players fetched...")
        time.sleep(0.05)

    history = pd.concat(all_histories, ignore_index=True) if all_histories else pd.DataFrame()
    print(f"  Fetched {len(history)} records across {history['player_id'].nunique()} players")
    save_cache(bootstrap, fixtures_raw, history, last_finished_gw)

# Standardise column names and types
history["player_id"] = pd.to_numeric(history["player_id"], errors="coerce")
history["round"]     = pd.to_numeric(history.get("round", history.get("event", np.nan)), errors="coerce")
history             = history.rename(columns={"round": "event"})
history["event"]    = pd.to_numeric(history["event"], errors="coerce")

# Exact stats from element-summary
EXACT_STATS = [
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded"
]
for col in EXACT_STATS:
    if col in history.columns:
        history[col] = pd.to_numeric(history[col], errors="coerce").fillna(0)
    else:
        history[col] = 0

# Merge fixture context (FDR, home/away) from fixtures table
fix_context = []
for _, fix in fixtures[finished_mask].iterrows():
    gw     = fix["event"]
    diff_h = float(fix.get("team_h_difficulty") or 3)
    diff_a = float(fix.get("team_a_difficulty") or 3)
    score_h = float(fix.get("team_h_score") or 0)
    score_a = float(fix.get("team_a_score") or 0)
    fix_context.append({"team": fix["team_h"], "event": gw, "fdr": diff_h,
                         "is_home": 1, "score_for": score_h, "score_ag": score_a})
    fix_context.append({"team": fix["team_a"], "event": gw, "fdr": diff_a,
                         "is_home": 0, "score_for": score_a, "score_ag": score_h})

fix_ctx_df = pd.DataFrame(fix_context)

# Merge team onto history via players table, then merge fixture context
player_team = players[["id","team"]].rename(columns={"id":"player_id"})
history = history.merge(player_team, on="player_id", how="left")
history = history.merge(fix_ctx_df, on=["team","event"], how="left")
history["fdr"]      = history["fdr"].fillna(3)
history["is_home"]  = history["is_home"].fillna(0)
history["score_for"]= history["score_for"].fillna(0)
history["score_ag"] = history["score_ag"].fillna(0)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Merge player metadata onto history
# ─────────────────────────────────────────────────────────────────────────────
META_COLS = [c for c in [
    "id", "element_type", "now_cost", "selected_by_percent",
    "expected_goals_per_90", "expected_assists_per_90",
    "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
    "goals_conceded_per_90", "saves_per_90", "clean_sheets_per_90",
    "web_name", "first_name", "second_name", "team",
    "minutes", "starts", "chance_of_playing_next_round", "status", "news"
] if c in players.columns]

history = history.merge(
    players[META_COLS], left_on="player_id", right_on="id", how="left",
    suffixes=("", "_season")
)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Build complete player × GW grid (zeros for missed GWs)
#    Every player gets a row for EVERY finished GW.
#    If they didn't play, all stats = 0, minutes = 0.
# ─────────────────────────────────────────────────────────────────────────────
print("Building player × GW grid with exact data...")

grid = pd.MultiIndex.from_product(
    [all_player_ids, finished_gws_sorted],
    names=["player_id", "event"]
).to_frame(index=False)

history["player_id"] = history["player_id"].astype(int)
history["event"]     = history["event"].astype(int)

GRID_STATS = [
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
    "yellow_cards", "red_cards", "saves", "bonus", "bps",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "score_for", "score_ag", "fdr", "is_home"
]
GRID_STATS = [c for c in GRID_STATS if c in history.columns]

grid = grid.merge(
    history[["player_id","event"] + GRID_STATS].drop_duplicates(["player_id","event"]),
    on=["player_id","event"], how="left"
)

FILL_ZERO = [c for c in GRID_STATS if c not in ("fdr","is_home")]
for col in FILL_ZERO:
    grid[col] = grid[col].fillna(0)
grid["fdr"]     = grid["fdr"].fillna(3)
grid["is_home"] = grid["is_home"].fillna(0)
grid = grid.sort_values(["player_id","event"]).reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Rolling 5-GW features  (shift(1) prevents leakage)
# ─────────────────────────────────────────────────────────────────────────────
print("Computing rolling features...")

ROLL_STATS = [
    "total_points", "minutes", "goals_scored", "assists", "clean_sheets",
    "goals_conceded", "saves", "bonus", "bps",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "score_for", "score_ag"
]
ROLL_STATS = [c for c in ROLL_STATS if c in grid.columns]

for stat in ROLL_STATS:
    grp = grid.groupby("player_id")[stat]
    grid[f"{stat}_mean5"] = grp.transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    grid[f"{stat}_sum5"]  = grp.transform(lambda x: x.shift(1).rolling(5, min_periods=1).sum())
    grid[f"{stat}_std5"]  = grp.transform(lambda x: x.shift(1).rolling(5, min_periods=1).std().fillna(0))

# Form trend: recent 5-GW avg vs earlier 10-GW avg (positive = improving)
grid["pts_trend"] = (
    grid["total_points_mean5"] -
    grid.groupby("player_id")["total_points"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=1).mean()
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Merge player metadata + position-specific features
# ─────────────────────────────────────────────────────────────────────────────
PER90_FEATURES = [c for c in [
    "expected_goals_per_90", "expected_assists_per_90",
    "expected_goal_involvements_per_90", "expected_goals_conceded_per_90",
    "goals_conceded_per_90", "saves_per_90", "clean_sheets_per_90"
] if c in players.columns]

grid = grid.merge(
    players[["id","element_type","now_cost","selected_by_percent"] + PER90_FEATURES],
    left_on="player_id", right_on="id", how="left"
)

# Interaction features
grid["pts_x_fdr"]      = grid["total_points_mean5"] * (6 - grid["fdr"])
grid["home_x_form"]    = grid["is_home"] * grid["total_points_mean5"]

# Position-specific features
# GK/DEF: clean sheet form is the key signal
grid["cs_form"]        = grid["clean_sheets_mean5"] * grid["element_type"].isin([1,2]).astype(int)
# MID/FWD: goal involvement form
grid["gi_form"]        = grid["expected_goal_involvements_mean5"] * grid["element_type"].isin([3,4]).astype(int)
# GK: saves form
grid["saves_form"]     = grid["saves_mean5"] * (grid["element_type"] == 1).astype(int)

# ─────────────────────────────────────────────────────────────────────────────
# 8. Define feature sets
#    Common features used by all position models.
#    Each position model also gets its own specialist features.
# ─────────────────────────────────────────────────────────────────────────────
COMMON_ROLL = (
    [f"{s}_mean5" for s in ROLL_STATS] +
    [f"{s}_sum5"  for s in ROLL_STATS] +
    [f"{s}_std5"  for s in ROLL_STATS]
)
COMMON_META = [
    "fdr", "is_home", "now_cost", "pts_x_fdr", "home_x_form", "pts_trend"
] + PER90_FEATURES

# Position-specific extra features
POS_EXTRA = {
    1: ["cs_form", "saves_form", "clean_sheets_per_90", "saves_per_90"],          # GK
    2: ["cs_form", "clean_sheets_per_90", "goals_conceded_per_90"],                # DEF
    3: ["gi_form", "expected_goals_per_90", "expected_assists_per_90"],            # MID
    4: ["gi_form", "expected_goals_per_90", "expected_assists_per_90"],            # FWD
}

BASE_FEATURES = COMMON_ROLL + COMMON_META
TARGET        = "total_points"
MIN_MINUTES   = 1   # avg min/GW in rolling window to be included

# ─────────────────────────────────────────────────────────────────────────────
# 9. Train one model per position  +  validate on 20% GWs
# ─────────────────────────────────────────────────────────────────────────────
print("\nTraining per-position XGBoost models...")

# Drop first GW and apply active player filter
model_df = grid[grid["event"] > finished_gws_sorted[0]].copy()
model_df = model_df[model_df["minutes_mean5"] > MIN_MINUTES]
model_df = model_df.dropna(subset=BASE_FEATURES + [TARGET])

POS_LABELS = {1:"GK", 2:"DEF", 3:"MID", 4:"FWD"}
models     = {}       # pos → fitted model
pos_maes   = {}       # pos → validation MAE
val_frames = []       # collect val predictions across positions

overall_train_rows = 0
overall_val_rows   = 0

for pos, pos_label in POS_LABELS.items():
    pos_df = model_df[model_df["element_type"] == pos].copy()
    if pos_df.empty:
        print(f"  {pos_label}: no data — skipping")
        continue

    # Features = base + position-specific extras
    # Use dict.fromkeys() to deduplicate while preserving order —
    # duplicate column names cause XGBoost to receive a DataFrame
    # instead of a Series, triggering the AttributeError on .dtype
    extra = [c for c in POS_EXTRA.get(pos, []) if c in pos_df.columns]
    feats = list(dict.fromkeys(
        c for c in BASE_FEATURES + extra if c in pos_df.columns
    ))

    train_p = pos_df[pos_df["event"].isin(train_gws)].dropna(subset=feats + [TARGET])
    val_p   = pos_df[pos_df["event"].isin(val_gws)].dropna(subset=feats + [TARGET])

    X_tr = train_p[feats].astype(float)
    y_tr = train_p[TARGET].astype(float)
    X_v  = val_p[feats].astype(float)
    y_v  = val_p[TARGET].astype(float)

    overall_train_rows += len(train_p)
    overall_val_rows   += len(val_p)

    m = XGBRegressor(
        n_estimators=600, learning_rate=0.04, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        objective="reg:squarederror", random_state=42, verbosity=0
    )

    if len(X_v) > 0:
        m.fit(X_tr, y_tr, eval_set=[(X_v, y_v)], verbose=False)
        val_preds      = m.predict(X_v).clip(min=0)
        mae            = mean_absolute_error(y_v, val_preds)
        pos_maes[pos]  = mae

        # Store per-player val predictions
        vf = val_p[["player_id","event",TARGET]].copy()
        vf["predicted"]    = val_preds
        vf["error"]        = (vf["predicted"] - vf[TARGET]).abs()
        vf["element_type"] = pos
        val_frames.append(vf)

        print(f"  {pos_label:<3}  train={len(train_p):,}  val={len(val_p):,}  MAE={mae:.3f} pts")
    else:
        m.fit(X_tr, y_tr, verbose=False)
        print(f"  {pos_label:<3}  train={len(train_p):,}  val=0  (no val GWs yet)")

    # Retrain on ALL data for final predictions
    all_p = pos_df.dropna(subset=feats + [TARGET])
    m.fit(all_p[feats].astype(float), all_p[TARGET].astype(float), verbose=False)

    models[pos] = {"model": m, "features": feats, "label": pos_label}

n_train_rows = overall_train_rows
n_val_rows   = overall_val_rows
total_rows   = n_train_rows + n_val_rows
train_pct    = 100 * n_train_rows / total_rows if total_rows else 0
val_pct      = 100 * n_val_rows   / total_rows if total_rows else 0

print(f"\n  ── Overall split ────────────────────────────────────")
print(f"  Train rows : {n_train_rows:,}  ({train_pct:.1f}%)")
print(f"  Val rows   : {n_val_rows:,}  ({val_pct:.1f}%)")
if pos_maes:
    overall_mae = np.mean(list(pos_maes.values()))
    print(f"  Overall MAE (avg across positions): {overall_mae:.3f} pts")
print(f"  ────────────────────────────────────────────────────")

# ─────────────────────────────────────────────────────────────────────────────
# 10. Per-player validation summary
# ─────────────────────────────────────────────────────────────────────────────
val_summary_out   = None
val_breakdown_out = None

if val_frames:
    val_all = pd.concat(val_frames, ignore_index=True)
    player_names = players[["id","web_name"]].copy()
    player_names["id"] = pd.to_numeric(player_names["id"], errors="coerce")

    per_player_val = (
        val_all.groupby("player_id")
        .agg(
            actual_avg   = (TARGET,        "mean"),
            predicted_avg= ("predicted",   "mean"),
            mae          = ("error",       "mean"),
            gws_eval     = ("event",       "count"),
            element_type = ("element_type","first"),
        )
        .reset_index()
        .merge(player_names, left_on="player_id", right_on="id", how="left")
        .rename(columns={
            "web_name"    : "Player",
            "element_type": "Pos",
            "actual_avg"  : "AvgActualPts",
            "predicted_avg":"AvgPredPts",
            "mae"         : "MAE",
            "gws_eval"    : "GWsEval",
        })
    )
    per_player_val["Pos"]          = per_player_val["Pos"].map(POS_LABELS).fillna("?")
    per_player_val["AvgActualPts"] = per_player_val["AvgActualPts"].round(2)
    per_player_val["AvgPredPts"]   = per_player_val["AvgPredPts"].round(2)
    per_player_val["MAE"]          = per_player_val["MAE"].round(3)

    val_summary_out = os.path.join("data", "predictions", f"fpl_validation_summary_gw{predict_gw}.csv")
    per_player_val[["Player","Pos","GWsEval","AvgActualPts","AvgPredPts","MAE"]].to_csv(
        val_summary_out, index=False
    )
    print(f"\n  Per-player validation saved → {val_summary_out}")

# ─────────────────────────────────────────────────────────────────────────────
# 11. Build GW fixture context for predict_gw
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nBuilding predictions for GW {predict_gw}...")

gw_fix  = fixtures[fixtures["event"] == predict_gw].copy()
home_gw = gw_fix[["team_h","team_h_difficulty"]].rename(columns={"team_h":"team","team_h_difficulty":"fdr"})
home_gw["is_home"] = 1
away_gw = gw_fix[["team_a","team_a_difficulty"]].rename(columns={"team_a":"team","team_a_difficulty":"fdr"})
away_gw["is_home"] = 0

gw_teams = pd.concat([home_gw, away_gw], ignore_index=True)
gw_teams["num_fixtures"] = gw_teams.groupby("team")["fdr"].transform("count")
gw_teams = gw_teams.sort_values("fdr").drop_duplicates("team")

# ─────────────────────────────────────────────────────────────────────────────
# 12. Rolling state at last_finished_gw (the feature snapshot for prediction)
# ─────────────────────────────────────────────────────────────────────────────
roll_cols = (
    [f"{s}_mean5" for s in ROLL_STATS] +
    [f"{s}_sum5"  for s in ROLL_STATS] +
    [f"{s}_std5"  for s in ROLL_STATS] +
    ["pts_trend","cs_form","gi_form","saves_form"]
)
roll_cols = [c for c in roll_cols if c in grid.columns]

latest = grid[grid["event"] == last_finished_gw][["player_id"] + roll_cols].copy()

# Fallback for players whose team had a blank in last_finished_gw
missing = set(all_player_ids) - set(latest["player_id"].unique())
if missing:
    fallback = (
        grid[grid["player_id"].isin(missing)]
        .sort_values("event").groupby("player_id").last().reset_index()
        [["player_id"] + roll_cols]
    )
    latest = pd.concat([latest, fallback], ignore_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# 13. Minutes-based availability score  (exact minutes now available)
# ─────────────────────────────────────────────────────────────────────────────
latest["avg_minutes_last5"] = latest["minutes_sum5"] / 5

def minutes_to_availability(avg_min):
    if avg_min >= 75: return 1.00
    if avg_min >= 60: return 0.90
    if avg_min >= 45: return 0.75
    if avg_min >= 30: return 0.55
    if avg_min >= 15: return 0.35
    return 0.10

latest["availability_score"] = latest["avg_minutes_last5"].apply(minutes_to_availability)

# ─────────────────────────────────────────────────────────────────────────────
# 14. Assemble prediction dataframe and predict per position
# ─────────────────────────────────────────────────────────────────────────────
pred_df = players[META_COLS].copy()
pred_df = pred_df.merge(latest, left_on="id", right_on="player_id", how="left")
pred_df = pred_df.merge(gw_teams[["team","fdr","is_home","num_fixtures"]], on="team", how="left")

pred_df["fdr"]               = pd.to_numeric(pred_df["fdr"],          errors="coerce").fillna(3)
pred_df["is_home"]            = pd.to_numeric(pred_df["is_home"],      errors="coerce").fillna(0)
pred_df["num_fixtures"]       = pd.to_numeric(pred_df["num_fixtures"], errors="coerce").fillna(0)
pred_df["avg_minutes_last5"]  = pred_df["avg_minutes_last5"].fillna(0)
pred_df["availability_score"] = pred_df["availability_score"].fillna(0.10)

for col in roll_cols + PER90_FEATURES + ["now_cost","pts_x_fdr","home_x_form","pts_trend",
                                          "cs_form","gi_form","saves_form"]:
    if col not in pred_df.columns:
        pred_df[col] = 0.0
    pred_df[col] = pd.to_numeric(pred_df[col], errors="coerce").fillna(0)

pred_df["pts_x_fdr"]   = pred_df["total_points_mean5"] * (6 - pred_df["fdr"])
pred_df["home_x_form"] = pred_df["is_home"] * pred_df["total_points_mean5"]

# Predict using each position's model
pred_df["raw_predicted_points"] = 0.0

for pos, info in models.items():
    mask  = pred_df["element_type"] == pos
    feats = info["features"]

    # ensure all required feature columns exist
    for c in feats:
        if c not in pred_df.columns:
            pred_df[c] = 0.0
        pred_df[c] = pd.to_numeric(pred_df[c], errors="coerce").fillna(0)

    feats = list(dict.fromkeys(feats))   # deduplicate just in case
    X_p = pred_df.loc[mask, feats].astype(float)
    if len(X_p) > 0:
        pred_df.loc[mask, "raw_predicted_points"] = info["model"].predict(X_p).clip(min=0)

pred_df["raw_predicted_points"] = pred_df["raw_predicted_points"].round(2)
pred_df.loc[pred_df["num_fixtures"] == 0, "raw_predicted_points"] = 0.0

pred_df["predicted_points"] = (
    pred_df["raw_predicted_points"] * pred_df["availability_score"]
).round(2)

pred_df["price_m"] = pred_df["now_cost"] / 10
pred_df["value"]   = (pred_df["predicted_points"] / pred_df["price_m"].clip(lower=0.1)).round(3)

# ─────────────────────────────────────────────────────────────────────────────
# 15. Debug trace
# ─────────────────────────────────────────────────────────────────────────────
if args.debug:
    mask = pred_df["web_name"].str.contains(args.debug, case=False, na=False)
    if mask.any():
        r = pred_df[mask].iloc[0]
        pos_label = POS_LABELS.get(int(r.get("element_type", 3)), "?")
        print(f"\n{'─'*60}")
        print(f"  PLAYER TRACE : {r['web_name']}  ({pos_label})")
        print(f"{'─'*60}")
        print(f"  Rolling window        : GWs {finished_gws_sorted[-5:]}")
        print(f"  avg_minutes_last5     : {r['avg_minutes_last5']:.1f} min/GW  (EXACT)")
        print(f"  availability_score    : {r['availability_score']}")
        print(f"  total_points_mean5    : {r['total_points_mean5']:.2f}  (EXACT FPL pts)")
        print(f"  goals_scored_mean5    : {r['goals_scored_mean5']:.2f}")
        print(f"  assists_mean5         : {r['assists_mean5']:.2f}")
        print(f"  clean_sheets_mean5    : {r['clean_sheets_mean5']:.2f}")
        print(f"  fdr                   : {r['fdr']}")
        print(f"  is_home               : {r['is_home']}")
        print(f"  num_fixtures          : {r['num_fixtures']}")
        print(f"  raw_predicted_points  : {r['raw_predicted_points']}")
        print(f"  × availability_score  : {r['availability_score']}")
        print(f"  = predicted_points    : {r['predicted_points']}")
        print(f"{'─'*60}")
        pid   = int(r["id"])
        last5 = grid[
            (grid["player_id"] == pid) &
            (grid["event"].isin(finished_gws_sorted[-5:]))
        ][["event","total_points","minutes","goals_scored","assists","clean_sheets","bonus"]]
        last5.columns = ["GW","Pts","Min","GS","Ast","CS","Bon"]
        print(f"\n  Last 5 GW breakdown (EXACT from API):")
        print(last5.to_string(index=False))
        print()
    else:
        print(f"\n  Debug: '{args.debug}' not found")

# ─────────────────────────────────────────────────────────────────────────────
# 16. Output — all players + best per position
# ─────────────────────────────────────────────────────────────────────────────
DISPLAY_COLS = [c for c in [
    "web_name","element_type","price_m","fdr","is_home","num_fixtures",
    "avg_minutes_last5","availability_score",
    "raw_predicted_points","predicted_points","value",
    "total_points_mean5","selected_by_percent"
] if c in pred_df.columns]

results = (
    pred_df[DISPLAY_COLS]
    .sort_values("predicted_points", ascending=False)
    .rename(columns={
        "web_name"             : "Player",
        "element_type"         : "Pos",
        "price_m"              : "Price(£m)",
        "fdr"                  : "FDR",
        "is_home"              : "Home",
        "num_fixtures"         : "Fixtures",
        "avg_minutes_last5"    : "AvgMin(L5)",
        "availability_score"   : "AvailScore",
        "raw_predicted_points" : "RawPts",
        "predicted_points"     : "PredPts",
        "value"                : "Value",
        "total_points_mean5"   : "AvgPts(L5)",
        "selected_by_percent"  : "Sel%",
    })
)
results["Pos"] = results["Pos"].map(POS_LABELS).fillna("?")

# ── Best players per position ────────────────────────────────────────────────
TOP_N = 10
best_by_pos = []
for pos_label in ["GK","DEF","MID","FWD"]:
    top = results[results["Pos"] == pos_label].head(TOP_N).copy()
    top.insert(0, "Rank", range(1, len(top)+1))
    best_by_pos.append(top)

print(f"\n{'='*80}")
print(f"  BEST PLAYERS BY POSITION — GW {predict_gw}")
print(f"{'='*80}")
for pos_label, top in zip(["GK","DEF","MID","FWD"], best_by_pos):
    pos_mae_str = f"  (pos MAE: {pos_maes[{'GK':1,'DEF':2,'MID':3,'FWD':4}[pos_label]]:.3f} pts)" \
                  if {'GK':1,'DEF':2,'MID':3,'FWD':4}[pos_label] in pos_maes else ""
    print(f"\n  ── {pos_label}{pos_mae_str} {'─'*50}")
    print(top[["Rank","Player","Price(£m)","FDR","Home","Fixtures",
               "AvgMin(L5)","AvailScore","RawPts","PredPts","Value"]].to_string(index=False))

# Save all predictions
pred_out = os.path.join("data", "predictions", f"fpl_predictions_gw{predict_gw}.csv")
results.to_csv(pred_out, index=False)

# Save best by position
best_out  = os.path.join("data", "predictions", f"fpl_best_by_position_gw{predict_gw}.csv")
pd.concat(best_by_pos).to_csv(best_out, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 17. Training summary
# ─────────────────────────────────────────────────────────────────────────────
summary_path = os.path.join("data", "predictions", "fpl_training_summary.txt")
with open(summary_path, "w") as f:
    f.write("FPL Model Training Summary\n==========================\n\n")
    f.write(f"Predicting GW     : {predict_gw}\n")
    f.write(f"Data source       : FPL API — exact per-GW history from element-summary\n\n")
    f.write(f"Finished GWs      : GW{finished_gws_sorted[0]}–GW{last_finished_gw} ({n_gws} total)\n")
    f.write(f"Train GWs (80%)   : GW{train_gws[0]}–GW{train_gws[-1]} ({len(train_gws)} GWs)\n")
    if val_gws:
        f.write(f"Val GWs   (20%)   : GW{val_gws[0]}–GW{val_gws[-1]} ({len(val_gws)} GWs)\n\n")
    f.write(f"Active filter     : avg minutes > {MIN_MINUTES} min/GW in rolling window\n")
    f.write(f"Train rows        : {n_train_rows:,} ({train_pct:.1f}%)\n")
    f.write(f"Val rows          : {n_val_rows:,} ({val_pct:.1f}%)\n\n")
    f.write(f"Per-position MAE on validation GWs:\n")
    for pos, label in POS_LABELS.items():
        mae_str = f"{pos_maes[pos]:.3f} pts" if pos in pos_maes else "n/a"
        f.write(f"  {label:<4}: {mae_str}\n")
    if pos_maes:
        f.write(f"  Overall avg: {np.mean(list(pos_maes.values())):.3f} pts\n")

print(f"\n✅  All players      → {pred_out}")
print(f"✅  Best by position → {best_out}")
if val_summary_out:
    print(f"✅  Validation       → {val_summary_out}")
print(f"✅  Summary          → {summary_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Export to S3  (only runs when --export --s3-bucket <name> is passed)
#
# What gets uploaded:
#   predictions/fpl_predictions_gw<N>.csv        ← chatbot reads this
#   predictions/fpl_best_by_position_gw<N>.csv   ← chatbot reads this
#   predictions/fpl_training_summary.txt
#   cache/bootstrap.json                          ← cache for next run
#   cache/fixtures.json
#   cache/player_history.parquet
#   cache/meta.json
#
# Lambda / chatbot only ever reads from predictions/ — it never touches
# the model code or the cache/ prefix.
# ─────────────────────────────────────────────────────────────────────────────
if args.export:
    if not args.s3_bucket:
        print("\n⚠️  --export requires --s3-bucket <bucket-name>")
    else:
        bucket = args.s3_bucket
        gw_tag = f"gw{predict_gw}"
        print(f"\nUploading to s3://{bucket} ...")

        # Prediction outputs — Lambda reads these
        upload_file_to_s3(pred_out,     f"predictions/fpl_predictions_{gw_tag}.csv",       bucket)
        upload_file_to_s3(best_out,     f"predictions/fpl_best_by_position_{gw_tag}.csv",  bucket)
        upload_file_to_s3(summary_path, f"predictions/fpl_training_summary.txt",           bucket)
        if val_summary_out:
            upload_file_to_s3(val_summary_out,
                              f"predictions/fpl_validation_{gw_tag}.csv", bucket)

        # Cache files — next local run can pull these down instead of re-fetching
        with open(CACHE_BOOTSTRAP) as f:
            bs_obj = json.load(f)
        with open(CACHE_FIXTURES) as f:
            fx_obj = json.load(f)
        upload_json_to_s3(bs_obj,  "cache/bootstrap.json",           bucket)
        upload_json_to_s3(fx_obj,  "cache/fixtures.json",            bucket)
        upload_df_to_s3(history,   "cache/player_history.parquet",   bucket, fmt="parquet")
        upload_json_to_s3({"last_finished_gw": last_finished_gw},
                          "cache/meta.json", bucket)

        print(f"\n✅  All files uploaded to s3://{bucket}")
        print(f"   Lambda chatbot reads: predictions/fpl_best_by_position_{gw_tag}.csv")

print(f"\nRun again next GW — API data updates automatically each week.")
print(f"To trace a player : python fpl_predictor.py --debug \"Salah\"")
print(f"To upload to S3   : python fpl_predictor.py --export --s3-bucket your-bucket-name")