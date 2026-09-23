import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import pandas as pd
import yfinance as yf
from strategy import analyze, chart_data, find_surge_event, evaluate_event

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

def fetch(t):
    d = yf.download(t, period="30d", interval="1d", auto_adjust=False,
                    progress=False, threads=False)
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    return d.dropna(subset=["Open", "High", "Low", "Close", "Volume"])

def fetch_and_analyze(t):
    try:
        d = fetch(t)
        r = analyze(d)
        if r:
            r["ticker"] = t
            r["chart"] = chart_data(d, 30)
            return r
    except Exception as e:
        print("scan error", t, e)
    return None

symbols = universe()
old_watch = load_watchlist()
watch_keys = {(x.get("ticker"), x.get("event_date")) for x in old_watch}
new_watch = []

rows = []
completed = 0
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(fetch_and_analyze, t): t for t in symbols}
    for future in as_completed(futures):
        completed += 1
        result = future.result()
        if result:
            rows.append(result)
            key = (result["ticker"], result["event_date"])
            if key not in watch_keys:
                new_watch.append(result)
        if completed % 100 == 0:
            print("scanned", completed, "of", len(symbols))

# Preserve candidates discovered on previous days and add newly discovered Day-0 events.
# Each candidate expires automatically after Day 20.
combined = {}
for item in old_watch + new_watch:
    key = (item.get("ticker"), item.get("event_date"))
    if key[0] and key[1]:
        combined[key] = item

active_watch = []
for item in combined.values():
    try:
        d = fetch(item["ticker"])
        event = find_surge_event(d)
        if not event or str(event["event_date"]) != str(item["event_date"]):
            continue
        evaluated = evaluate_event(d, event)
        if evaluated is None:
            continue
        evaluated["ticker"] = item["ticker"]
        evaluated["chart"] = chart_data(d, 30)
        active_watch.append(evaluated)
    except Exception as e:
        print("watch error", item.get("ticker"), e)

save_watchlist(active_watch)

# Dashboard shows only active 20-day candidates.
rows = active_watch
order = {"جاهز للدخول": 3, "شبه جاهز": 2, "قيد المراقبة": 1}
rows.sort(key=lambda x: (-order.get(x["status"], 0),
                         -x["technical"]["positive_confirmations"],
                         -x["surge_pct"]))

payload = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "count": len(rows),
    "stats": {
        "ready": sum(x["status"] == "جاهز للدخول" for x in rows),
        "semi_ready": sum(x["status"] == "شبه جاهز" for x in rows),
        "watching": sum(x["status"] == "قيد المراقبة" for x in rows)
    },
    "signals": rows
}
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)
print("active watchlist", len(rows), "new events", len(new_watch))
