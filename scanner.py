import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
from strategy import analyze, chart_data, find_pattern_event, evaluate_event, technicals

WATCHLIST_FILE = "watchlist.json"


def universe():
    out = set()
    for url, col in [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol")
    ]:
        try:
            d = pd.read_csv(url, sep="|")
            d = d[d["Test Issue"] == "N"]
            out.update(str(x).strip().upper() for x in d[col].dropna())
        except Exception as e:
            print("universe error:", e)
    return sorted(x for x in out if x.isalpha() and len(x) <= 5)


def load_watchlist():
    try:
        with open(WATCHLIST_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("items", [])
    except Exception:
        return []


def save_watchlist(items):
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "items": items
        }, f, ensure_ascii=False, indent=2)


def fetch(ticker, interval="1d"):
    period = "30d" if interval == "1d" else "60d"
    d = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=False, progress=False, threads=False
    )
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    return d.dropna(subset=needed)


def scan_one(ticker):
    try:
        d = fetch(ticker, "1d")
        if d.empty:
            return None
        event = find_pattern_event(d)
        return ticker, d, event
    except Exception as e:
        print("scan error", ticker, e)
        return None


def stored_event(item, d):
    event_date = str(item.get("event_date", ""))
    matches = [i for i, idx in enumerate(d.index) if str(idx.date()) == event_date]
    if not matches:
        return None

    idx = matches[-1]
    pattern_type = item.get("pattern_type")
    if not pattern_type:
        return None

    pattern_low = item.get("pattern_low")
    left_low = item.get("left_low")
    right_low = item.get("right_low")
    head_low = item.get("head_low")
    neckline = item.get("neckline")
    height = item.get("pattern_height")

    if any(v is None for v in [pattern_low, left_low, right_low, neckline]):
        return None

    return {
        "event_idx": idx,
        "event_date": event_date,
        "event_price": float(item.get("event_price", d["Close"].iloc[idx])),
        "pattern_type": pattern_type,
        "pattern_low": float(pattern_low),
        "left_low": float(left_low),
        "head_low": float(head_low) if head_low is not None else None,
        "right_low": float(right_low),
        "neckline": float(neckline),
        "pattern_height": float(height if height is not None else float(neckline) - float(pattern_low)),
    }


symbols = universe()
old_watch = load_watchlist()
old_keys = {(x.get("ticker"), x.get("event_date")) for x in old_watch}

market_data = {}
new_watch = []

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(scan_one, t): t for t in symbols}
    completed = 0

    for future in as_completed(futures):
        completed += 1
        result = future.result()
        if result:
            ticker, d, event = result
            market_data[ticker] = (d, None)

            if event:
                key = (ticker, event["event_date"])
                if key not in old_keys:
                    item = dict(event)
                    item["ticker"] = ticker
                    item["event_date"] = event["event_date"]
                    item["event_price"] = round(event["event_price"], 4)
                    item["pattern_low"] = round(event["pattern_low"], 4)
                    item["left_low"] = round(event["left_low"], 4)
                    item["head_low"] = round(event["head_low"], 4) if event.get("head_low") is not None else None
                    item["right_low"] = round(event["right_low"], 4)
                    item["neckline"] = round(event["neckline"], 4)
                    item["pattern_height"] = round(event["pattern_height"], 4)
                    item["discovered_at"] = datetime.now(timezone.utc).isoformat()
                    new_watch.append(item)

        if completed % 100 == 0:
            print("scanned", completed, "of", len(symbols))

candidate_tickers = {x.get("ticker") for x in old_watch if x.get("ticker")}
candidate_tickers.update(x.get("ticker") for x in new_watch if x.get("ticker"))

# 4H only for pattern candidates and persistent watches.
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {
        pool.submit(fetch, ticker, "4h"): ticker
        for ticker in candidate_tickers
        if ticker in market_data
    }
    for future in as_completed(futures):
        ticker = futures[future]
        try:
            market_data[ticker] = (market_data[ticker][0], future.result())
        except Exception as e:
            print("4h error", ticker, e)
            market_data[ticker] = (market_data[ticker][0], None)


combined = {}
for item in old_watch + new_watch:
    key = (item.get("ticker"), item.get("event_date"))
    if key[0] and key[1]:
        combined[key] = item

active_watch = []

for key, item in combined.items():
    ticker = item.get("ticker")
    md = market_data.get(ticker)
    try:
        if md is None:
            continue

        d, h4 = md
        if d.empty:
            continue

        event = stored_event(item, d)
        if not event:
            continue

        h4tech = technicals(h4.tail(30)) if h4 is not None and not h4.empty else {}
        evaluated = evaluate_event(d, event, h4tech)
        if evaluated is None:
            continue

        evaluated["ticker"] = ticker
        evaluated["discovered_at"] = item.get("discovered_at")
        evaluated["chart"] = chart_data(d, 30)
        evaluated["technical_4h"] = h4tech

        if item.get("ready_since"):
            evaluated["ready_since"] = item["ready_since"]
        elif evaluated["status"] == "جاهز للدخول":
            evaluated["ready_since"] = datetime.now(timezone.utc).isoformat()

        active_watch.append(evaluated)

    except Exception as e:
        print("watch error", ticker, e)

save_watchlist(active_watch)

order = {"جاهز للدخول": 3, "شبه جاهز": 2, "قيد المراقبة": 1}
active_watch.sort(key=lambda x: (
    -order.get(x["status"], 0),
    -x.get("readiness_score", 0),
    x.get("watch_age", 0)
))

payload = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "mode": "pattern_trial_10_days",
    "count": len(active_watch),
    "stats": {
        "ready": sum(x["status"] == "جاهز للدخول" for x in active_watch),
        "semi_ready": sum(x["status"] == "شبه جاهز" for x in active_watch),
        "watching": sum(x["status"] == "قيد المراقبة" for x in active_watch)
    },
    "signals": active_watch
}

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print("pattern trial active watchlist", len(active_watch), "new patterns", len(new_watch))
