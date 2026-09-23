import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
from strategy import analyze, chart_data, find_surge_event, evaluate_event, technicals

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
        # Daily candles discover the prior rally/base event; 4-hour candles refine the current recovery.
        d = fetch(ticker, "1d")
        h4 = fetch(ticker, "4h")
        if d.empty:
            return None
        result = analyze(d)
        event = find_surge_event(d)
        return ticker, d, h4, result, event
    except Exception as e:
        print("scan error", ticker, e)
        return None


def stored_event(item, d):
    """Rebuild the exact stored Day-0 event from its persistent date/base/high."""
    event_date = str(item.get("event_date", ""))
    matches = [i for i, idx in enumerate(d.index) if str(idx.date()) == event_date]
    if not matches:
        return None

    idx = matches[-1]
    base = float(item.get("prior_base", item.get("base", 0)))
    high = float(item.get("prior_high", item.get("high", 0)))
    event_price = float(item.get("event_price", d["Close"].iloc[idx]))
    rally = float(item.get("prior_rally_pct", item.get("surge_pct", 0)))

    if base <= 0 or high <= 0:
        return None

    return {
        "event_idx": idx,
        "event_date": event_date,
        "event_price": event_price,
        "base": base,
        "base_idx": max(0, idx - 19),
        "high": high,
        "high_idx": idx,
        "rally_pct": rally,
    }


symbols = universe()
old_watch = load_watchlist()
old_keys = {(x.get("ticker"), x.get("event_date")) for x in old_watch}

# One download per ticker per daily run. The same data is reused for discovery
# and for persistent-watchlist evaluation.
market_data = {}
new_watch = []

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(scan_one, t): t for t in symbols}
    completed = 0

    for future in as_completed(futures):
        completed += 1
        result = future.result()
        if result:
            ticker, d, h4, discovered, event = result
            market_data[ticker] = (d, h4)

            if event:
                key = (ticker, event["event_date"])
                if key not in old_keys:
                    item = dict(discovered) if discovered else {}
                    item["ticker"] = ticker
                    item["event_date"] = event["event_date"]
                    item["event_price"] = round(event["event_price"], 4)
                    item["prior_base"] = round(event["base"], 4)
                    item["prior_high"] = round(event["high"], 4)
                    item["prior_rally_pct"] = round(event["rally_pct"], 2)
                    item["discovered_at"] = datetime.now(timezone.utc).isoformat()
                    new_watch.append(item)

        if completed % 100 == 0:
            print("scanned", completed, "of", len(symbols))

# Merge by exact (ticker, Day-0 date). Existing events are never replaced by
# a newer rally in the same ticker; each event gets its own 20-day lifetime.
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

        evaluated = evaluate_event(d, event)
        if evaluated is None:
            continue

        # 4-hour technical snapshot is informational and does not override the daily entry gate.
        if h4 is not None and not h4.empty:
            evaluated["technical_4h"] = technicals(h4.tail(30))
        evaluated["ticker"] = ticker
        evaluated["discovered_at"] = item.get("discovered_at")
        evaluated["chart"] = chart_data(d, 30)

        # Preserve the first time this exact event became ready.
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
    -x["technical"]["positive_confirmations"],
    -x["surge_pct"]
))

payload = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
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

print("active watchlist", len(active_watch), "new events", len(new_watch))
