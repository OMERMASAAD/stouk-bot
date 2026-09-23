# -*- coding: utf-8 -*-
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from strategy import (
    STAGE_ORDER, STAGE_READY, STAGE_SEMI, STAGE_WATCH, MAX_FLOAT,
    chart_data, evaluate_event, find_surge_event, technicals,
)
from news import fetch_news

WATCHLIST_FILE = "watchlist.json"  # القائمة المستمرة تُحفظ مع كل تشغيل للرادار


def universe():
    out = set()
    for url, col in [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol"),
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
            "items": items,
        }, f, ensure_ascii=False, indent=2)


def fetch(ticker, interval="1d"):
    # 6 أشهر يومي: كافية لاكتشاف الصعود السابق والقاع وحساب EMA/RSI بدقة
    period = "6mo" if interval == "1d" else "60d"
    d = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=False, progress=False, threads=False,
    )
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    return d.dropna(subset=needed)


def get_float(ticker):
    """Float من Yahoo (floatShares، وإلا sharesOutstanding). None إذا غير متوفر."""
    try:
        info = yf.Ticker(ticker).info or {}
        f = info.get("floatShares") or info.get("sharesOutstanding")
        return int(f) if f else None
    except Exception as e:
        print("float error", ticker, e)
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


def scan_one(ticker, keep_tickers):
    try:
        d = fetch(ticker, "1d")
        if d.empty or len(d) < 40:
            return None
        event = find_surge_event(d)
        # تقييم أولي بدون Float (يُفحص Float لاحقًا للمرشحين فقط لتوفير الطلبات)
        result = evaluate_event(d, event) if event else None
        keep = d if (result or ticker in keep_tickers) else None
        return ticker, keep, result, event
    except Exception as e:
        print("scan error", ticker, e)
        return None


def main():
    symbols = universe()
    old_watch = load_watchlist()
    old_keys = {(x.get("ticker"), x.get("event_date")) for x in old_watch}
    keep_tickers = {x.get("ticker") for x in old_watch if x.get("ticker")}

    market_data = {}
    new_watch = []

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(scan_one, t, keep_tickers): t for t in symbols}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            res = future.result()
            if res:
                ticker, d, discovered, event = res
                if d is not None:
                    market_data[ticker] = (d, None)
                if discovered and event:
                    key = (ticker, event["event_date"])
                    if key not in old_keys:
                        item = dict(discovered)
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

    combined = {}
    for item in old_watch + new_watch:
        key = (item.get("ticker"), item.get("event_date"))
        if key[0] and key[1]:
            combined[key] = item

    # ---- Float (Yahoo) للمرشحين فقط ----
    need_float = sorted({k[0] for k, it in combined.items() if not it.get("float_shares")})
    floats = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        ff = {pool.submit(get_float, t): t for t in need_float}
        for f in as_completed(ff):
            floats[ff[f]] = f.result()

    # ---- فريم 4 ساعات للمرشحين (معلومة إضافية فقط) ----
    candidate_tickers = {k[0] for k in combined if k[0] in market_data}
    with ThreadPoolExecutor(max_workers=8) as pool:
        h4_futures = {pool.submit(fetch, t, "4h"): t for t in candidate_tickers}
        for future in as_completed(h4_futures):
            ticker = h4_futures[future]
            try:
                market_data[ticker] = (market_data[ticker][0], future.result())
            except Exception as e:
                print("4h error", ticker, e)

    # ---- إعادة تقييم كل الأحداث ----
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
            fl = item.get("float_shares") or floats.get(ticker)
            if fl is not None and fl > MAX_FLOAT:
                continue  # Float أكبر من 10M
            evaluated = evaluate_event(d, event, fl)
            if evaluated is None:
                continue
            evaluated["float_known"] = fl is not None
            if h4 is not None and not h4.empty:
                evaluated["technical_4h"] = technicals(h4.tail(30))
            evaluated["ticker"] = ticker
            evaluated["discovered_at"] = item.get("discovered_at")
            evaluated["chart"] = chart_data(d, 30)
            if item.get("ready_since"):
                evaluated["ready_since"] = item["ready_since"]
            elif evaluated["status"] == STAGE_READY:
                evaluated["ready_since"] = datetime.now(timezone.utc).isoformat()
            active_watch.append(evaluated)
        except Exception as e:
            print("watch error", ticker, e)

    # ---- طبقة الأخبار (Yahoo) بعد دخول الرادار ----
    with ThreadPoolExecutor(max_workers=6) as pool:
        nf = {pool.submit(fetch_news, x["ticker"]): x for x in active_watch}
        for f in as_completed(nf):
            x = nf[f]
            try:
                n = f.result()
            except Exception as e:
                print("news error", x["ticker"], e)
                n = {"warnings": [], "catalysts": [], "temp_excluded": False}
            x["news"] = n
            x["temp_excluded"] = bool(n.get("temp_excluded"))

    save_watchlist(active_watch)

    # الترتيب التشغيلي: المرحلة ثم درجة الجاهزية
    active_watch.sort(key=lambda x: (
        -STAGE_ORDER.get(x["status"], 0),
        -x["readiness_score"],
    ))

    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "count": len(active_watch),
        "stats": {
            "ready": sum(x["status"] == STAGE_READY for x in active_watch),
            "semi_ready": sum(x["status"] == STAGE_SEMI for x in active_watch),
            "watching": sum(x["status"] == STAGE_WATCH for x in active_watch),
            "warnings": sum(bool(x.get("news", {}).get("warnings")) for x in active_watch),
        },
        "signals": active_watch,
    }
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    print("active watchlist", len(active_watch), "new events", len(new_watch))


if __name__ == "__main__":
    main()
