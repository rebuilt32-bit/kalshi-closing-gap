"""
Closing-time gap analysis.

For markets in their final ~3 minutes, computes whether the price physically
has time to cross the strike at the current velocity.

Margin ratio = seconds_to_cross / seconds_left.
A 2x+ margin means the price would need 2x more time than remains
to cross the strike — so the currently-winning side is very likely to stay winning.

Modes:
  - Default (full): live calls + per-asset backtest + recent outcomes
  - LIVE_ONLY (env CLOSING_GAP_LIVE_ONLY=1): live calls only, fast path
"""
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

LIVE_ONLY = os.environ.get("CLOSING_GAP_LIVE_ONLY") == "1"
OUT_PATH = "data/closing_gap_live.json" if LIVE_ONLY else "data/closing_gap_analysis.json"
HIST_DIR = "data/history"
SETTLED_PATH = "data/settled.jsonl"

WINDOW_SECONDS = 240
VELOCITY_LOOKBACK_SECONDS = 60
RECENT_OUTCOMES_LIMIT = 10

BACKTEST_CHECKPOINTS = [
    {"label": "3min_left", "target_seconds": 165, "min": 130, "max": 200},
    {"label": "2min_left", "target_seconds": 110, "min": 80,  "max": 130},
    {"label": "1min_left", "target_seconds": 50,  "min": 20,  "max": 80},
]

BUCKETS = [
    {"label": "extreme",     "min": 3.0,  "max": float("inf")},
    {"label": "exceptional", "min": 2.5,  "max": 3.0},
    {"label": "very_high",   "min": 2.0,  "max": 2.5},
    {"label": "high",        "min": 1.75, "max": 2.0},
    {"label": "moderate",    "min": 1.5,  "max": 1.75},
    {"label": "narrow",      "min": 1.25, "max": 1.5},
    {"label": "coinflip",    "min": 1.0,  "max": 1.25},
    {"label": "losing_side", "min": -1.0, "max": 1.0},
]


def classify(margin_ratio):
    if margin_ratio is None:
        return None
    for b in BUCKETS:
        if b["min"] <= margin_ratio < b["max"]:
            return b["label"]
    return "losing_side"


def parse_iso(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def tail_jsonl(path, n=200):
    """Read last n lines of a jsonl file efficiently for LIVE_ONLY mode."""
    if not os.path.exists(path):
        return []
    size = os.path.getsize(path)
    read_size = min(size, 500000)
    with open(path, "rb") as f:
        f.seek(size - read_size)
        chunk = f.read().decode("utf-8", errors="ignore")
    lines = chunk.split("\n")
    if size > read_size and lines:
        lines = lines[1:]
    lines = lines[-n:]
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def composite_price(asset_data):
    prices = []
    if asset_data.get("kraken") is not None:
        prices.append(asset_data["kraken"])
    if asset_data.get("coinbase") is not None:
        prices.append(asset_data["coinbase"])
    if asset_data.get("binance_us") is not None:
        prices.append(asset_data["binance_us"])
    if not prices:
        return None
    return sum(prices) / len(prices)


def get_asset_series(history, asset_name):
    series = []
    for snap in history:
        a = snap.get("assets", {}).get(asset_name)
        if not a:
            continue
        cp = composite_price(a)
        if cp is None:
            continue
        ts = parse_iso(snap.get("ts"))
        if ts is None:
            continue
        series.append({"t": ts, "cp": cp})
    series.sort(key=lambda x: x["t"])
    return series


def velocity_from_series(series, ref_time, lookback_sec=VELOCITY_LOOKBACK_SECONDS):
    """Average abs velocity (price units per second) over last lookback_sec."""
    cutoff = ref_time.timestamp() - lookback_sec
    points = [
        p for p in series
        if cutoff <= p["t"].timestamp() <= ref_time.timestamp()
    ]
    if len(points) < 2:
        return None
    elapsed = points[-1]["t"].timestamp() - points[0]["t"].timestamp()
    if elapsed <= 0:
        return None
    delta = abs(points[-1]["cp"] - points[0]["cp"])
    return delta / elapsed


def margin_ratio_from(price, strike, seconds_left, velocity):
    gap = abs(price - strike)
    if seconds_left <= 0:
        return None, "expired"
    if velocity is None:
        return None, "no_velocity"
    if velocity <= 0:
        if gap > 0:
            return float("inf"), "extreme"
        return 0.0, "losing_side"
    seconds_to_cross = gap / velocity
    margin = seconds_to_cross / seconds_left
    return margin, classify(margin)


def load_all_history():
    if not os.path.exists(HIST_DIR):
        return []
    if LIVE_ONLY:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = os.path.join(HIST_DIR, today + ".jsonl")
        return tail_jsonl(path, 200)
    all_history = []
    for fname in sorted(os.listdir(HIST_DIR)):
        if fname.endswith(".jsonl"):
            all_history.extend(load_jsonl(os.path.join(HIST_DIR, fname)))
    return all_history


def compute_live(history):
    if not history:
        return []
    sorted_hist = sorted(
        history,
        key=lambda h: parse_iso(h.get("ts")) or datetime.min.replace(tzinfo=timezone.utc),
    )
    latest = sorted_hist[-1]
    now = parse_iso(latest.get("ts"))
    if now is None:
        return []

    series_by_asset = {}
    live_results = []

    for asset_name, asset_data in latest.get("assets", {}).items():
        if asset_name not in series_by_asset:
            series_by_asset[asset_name] = get_asset_series(history, asset_name)
        series = series_by_asset[asset_name]
        if not series:
            continue
        current_price = series[-1]["cp"]
        velocity = velocity_from_series(series, now)

        for market in asset_data.get("markets", []):
            if market.get("status") != "active":
                continue
            try:
                strike = float(market.get("strike"))
            except (TypeError, ValueError):
                continue
            close_time = parse_iso(market.get("close_time"))
            if close_time is None:
                continue
            seconds_left = (close_time - now).total_seconds()
            if seconds_left <= 0 or seconds_left > WINDOW_SECONDS:
                continue

            margin, bucket = margin_ratio_from(current_price, strike, seconds_left, velocity)

            if current_price > strike:
                safe_side = "YES"
                explanation = f"{asset_name} ${current_price - strike:.2f} above strike"
            elif current_price < strike:
                safe_side = "NO"
                explanation = f"{asset_name} ${strike - current_price:.2f} below strike"
            else:
                safe_side = None
                explanation = "exactly at strike"

            seconds_to_cross_val = None
            if velocity and velocity > 0:
                seconds_to_cross_val = round(abs(current_price - strike) / velocity, 1)

            margin_display = "infinite"
            if margin is not None and margin != float("inf"):
                margin_display = round(margin, 2)

            live_results.append({
                "ticker": market.get("ticker"),
                "asset": asset_name,
                "strike": strike,
                "current_price": round(current_price, 4),
                "gap": round(abs(current_price - strike), 4),
                "seconds_left": int(seconds_left),
                "velocity_per_sec": round(velocity, 6) if velocity is not None else None,
                "seconds_to_cross": seconds_to_cross_val,
                "margin_ratio": margin_display,
                "bucket": bucket,
                "safe_side": safe_side,
                "safe_explanation": explanation,
                "market_yes_bid": market.get("yes_bid"),
                "market_yes_ask": market.get("yes_ask"),
            })

    bucket_order = {b["label"]: i for i, b in enumerate(BUCKETS)}
    live_results.sort(key=lambda x: bucket_order.get(x["bucket"], 99))
    return live_results


def compute_backtest(history):
    settlements = load_jsonl(SETTLED_PATH)
    settled_by_ticker = {}
    for s in settlements:
        ticker = s.get("ticker")
        outcome = s.get("outcome")
        if ticker and outcome in ("YES", "NO"):
            settled_by_ticker[ticker] = outcome

    if not settled_by_ticker:
        return None

    ticker_info = {}
    for snap in history:
        snap_time = parse_iso(snap.get("ts"))
        if snap_time is None:
            continue
        for asset_name, asset_data in snap.get("assets", {}).items():
            cp = composite_price(asset_data)
            if cp is None:
                continue
            for market in asset_data.get("markets", []):
                ticker = market.get("ticker")
                if ticker not in settled_by_ticker:
                    continue
                try:
                    strike = float(market.get("strike"))
                except (TypeError, ValueError):
                    continue
                close_time = parse_iso(market.get("close_time"))
                if close_time is None:
                    continue
                if ticker not in ticker_info:
                    ticker_info[ticker] = {
                        "asset": asset_name,
                        "strike": strike,
                        "close_time": close_time,
                        "snapshots": [],
                    }
                seconds_left = (close_time - snap_time).total_seconds()
                ticker_info[ticker]["snapshots"].append({
                    "snap_time": snap_time,
                    "current_price": cp,
                    "seconds_left": seconds_left,
                })

    asset_series_cache = {}
    results = []
    ticker_strongest = {}
    bucket_strength = {b["label"]: i for i, b in enumerate(BUCKETS)}

    for ticker, info in ticker_info.items():
        outcome = settled_by_ticker[ticker]
        asset_name = info["asset"]
        if asset_name not in asset_series_cache:
            asset_series_cache[asset_name] = get_asset_series(history, asset_name)
        series = asset_series_cache[asset_name]

        for cp_def in BACKTEST_CHECKPOINTS:
            in_range = [
                s for s in info["snapshots"]
                if cp_def["min"] <= s["seconds_left"] <= cp_def["max"]
            ]
            if not in_range:
                continue
            best = min(in_range, key=lambda s: abs(s["seconds_left"] - cp_def["target_seconds"]))

            velocity = velocity_from_series(series, best["snap_time"])
            margin, bucket = margin_ratio_from(
                best["current_price"], info["strike"], best["seconds_left"], velocity
            )

            if best["current_price"] > info["strike"]:
                safe_side = "YES"
            elif best["current_price"] < info["strike"]:
                safe_side = "NO"
            else:
                safe_side = None

            won = None if safe_side is None else (safe_side == outcome)

            results.append({
                "ticker": ticker,
                "asset": asset_name,
                "checkpoint": cp_def["label"],
                "bucket": bucket,
                "safe_side_won": won,
            })

            if won is not None and bucket in bucket_strength:
                strength = bucket_strength[bucket]
                if (ticker not in ticker_strongest or
                        strength < ticker_strongest[ticker]["_strength"]):
                    ticker_strongest[ticker] = {
                        "ticker": ticker,
                        "asset": asset_name,
                        "close_time": info["close_time"].isoformat(),
                        "checkpoint": cp_def["label"],
                        "bucket": bucket,
                        "safe_side": safe_side,
                        "won": won,
                        "_strength": strength,
                    }

    by_bucket = defaultdict(lambda: {"n": 0, "wins": 0, "n_uncalled": 0})
    by_bucket_x_cp = defaultdict(lambda: defaultdict(lambda: {"n": 0, "wins": 0}))
    by_asset = defaultdict(lambda: {
        "by_bucket": defaultdict(lambda: {"n": 0, "wins": 0, "n_uncalled": 0}),
        "by_bucket_x_cp": defaultdict(lambda: defaultdict(lambda: {"n": 0, "wins": 0})),
    })

    for r in results:
        bucket = r["bucket"]
        cp_label = r["checkpoint"]
        won = r["safe_side_won"]
        asset = r["asset"]
        if won is None:
            by_bucket[bucket]["n_uncalled"] += 1
            by_asset[asset]["by_bucket"][bucket]["n_uncalled"] += 1
            continue
        by_bucket[bucket]["n"] += 1
        by_bucket_x_cp[bucket][cp_label]["n"] += 1
        by_asset[asset]["by_bucket"][bucket]["n"] += 1
        by_asset[asset]["by_bucket_x_cp"][bucket][cp_label]["n"] += 1
        if won:
            by_bucket[bucket]["wins"] += 1
            by_bucket_x_cp[bucket][cp_label]["wins"] += 1
            by_asset[asset]["by_bucket"][bucket]["wins"] += 1
            by_asset[asset]["by_bucket_x_cp"][bucket][cp_label]["wins"] += 1

    def make_bucket_summary(bb):
        out = {}
        for b in BUCKETS:
            label = b["label"]
            stats = bb.get(label, {"n": 0, "wins": 0, "n_uncalled": 0})
            n = stats["n"]
            out[label] = {
                "n_calls": n,
                "n_wins": stats["wins"],
                "win_rate": round(stats["wins"] / n, 3) if n > 0 else None,
                "n_at_strike_skipped": stats.get("n_uncalled", 0),
            }
        return out

    def make_bucket_x_cp(bxcp):
        out = {}
        for b in BUCKETS:
            label = b["label"]
            out[label] = {}
            for cp_def in BACKTEST_CHECKPOINTS:
                cp_label = cp_def["label"]
                stats = bxcp.get(label, {}).get(cp_label, {"n": 0, "wins": 0})
                n = stats["n"]
                out[label][cp_label] = {
                    "n": n,
                    "win_rate": round(stats["wins"] / n, 3) if n > 0 else None,
                }
        return out

    by_asset_out = {}
    for asset, ad in by_asset.items():
        by_asset_out[asset] = {
            "by_bucket": make_bucket_summary(ad["by_bucket"]),
            "by_bucket_x_checkpoint": make_bucket_x_cp(ad["by_bucket_x_cp"]),
        }

    recent_outcomes = defaultdict(list)
    for ticker, call in ticker_strongest.items():
        call.pop("_strength", None)
        recent_outcomes[call["asset"]].append(call)
    for asset in list(recent_outcomes.keys()):
        recent_outcomes[asset].sort(key=lambda x: x["close_time"], reverse=True)
        recent_outcomes[asset] = recent_outcomes[asset][:RECENT_OUTCOMES_LIMIT]

    return {
        "n_settled_tickers_evaluated": len(ticker_info),
        "n_total_calls": sum(by_bucket[b]["n"] for b in by_bucket),
        "by_bucket": make_bucket_summary(by_bucket),
        "by_bucket_x_checkpoint": make_bucket_x_cp(by_bucket_x_cp),
        "by_asset": by_asset_out,
        "recent_outcomes": dict(recent_outcomes),
    }


def main():
    history = load_all_history()
    if not history:
        result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "note": "No history yet.",
            "live_calls": [],
            "backtest": None,
        }
        os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print("No history yet.")
        return

    live_calls = compute_live(history)
    backtest = None if LIVE_ONLY else compute_backtest(history)

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "window_seconds": WINDOW_SECONDS,
            "velocity_lookback_seconds": VELOCITY_LOOKBACK_SECONDS,
            "live_only": LIVE_ONLY,
        },
        "interpretation": (
            "Margin ratio = seconds_to_cross / seconds_left. "
            ">1.0 means price needs more time than remains to cross the strike. "
            "Higher = safer for the currently-winning side."
        ),
        "live_calls": live_calls,
        "backtest": backtest,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    mode = "LIVE_ONLY" if LIVE_ONLY else "full"
    print(f"Closing-gap analysis ({mode}): {len(live_calls)} live calls in final {WINDOW_SECONDS}s window")
    for c in live_calls[:10]:
        m = c["margin_ratio"]
        m_str = f"{m}x" if m != "infinite" else "inf"
        print(f"  {c['asset']:5s} {c['ticker'][:30]:30s} {c['seconds_left']:>4d}s  "
              f"gap=${c['gap']:.2f}  margin={m_str:>7s} ({c['bucket']:<11s}) safe={c['safe_side']}")
    if backtest:
        print()
        print(f"Backtest: {backtest['n_total_calls']} calls across {backtest['n_settled_tickers_evaluated']} tickers")
        for b in BUCKETS:
            label = b["label"]
            stats = backtest["by_bucket"].get(label, {})
            n = stats.get("n_calls", 0)
            wr = stats.get("win_rate")
            wr_str = f"{wr:.3f}" if wr is not None else "n/a"
            print(f"  {label:14s} n={n:>5d}  win_rate={wr_str}")


if __name__ == "__main__":
    main()
