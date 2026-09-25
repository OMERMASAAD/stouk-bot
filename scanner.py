# -*- coding: utf-8 -*-
"""
الفحص اليومي لرادار القاع (Bottom Radar — Second Leg Setup).

خطوات الفحص:
  1) تحميل قائمة الرموز من NASDAQ/NYSE/AMEX.
  2) تحميل الشموع اليومية لكل رمز واختيار من يحقق: السعر $1–$5، وصعود سابق +100%
     خلال نافذة 20 جلسة، وDays_Since_Peak <= 20 (فلتر انتهاء الصلاحية).
  3) للمرشحين فقط: جلب Float (يُستبعد إذا كان > 10M أو غير متوفر)، وشموع 1h/4h
     لاكتشاف نمط القاع، والأخبار + مواعيد النتائج (تحذيرات/محفزات).
  4) حساب درجة الجاهزية (0–100) والمرحلة، وتوليد خطة الصفقة عند «جاهز فنيًا».
  5) كتابة data.json (للداشبورد) وwatchlist.json (حالة مستمرة: Float، الإضافة، الجاهزية).

ملاحظة تشغيلية: yfinance لا يدعم فريم 4h مباشرة، لذلك نجلب 1h ونعيد تجميعها إلى 4h.
"""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import pandas as pd
import yfinance as yf

import float_lookup
from news import fetch_news
from strategy import (
    MAX_DAYS_SINCE_PEAK, MAX_FLOAT, MAX_PRICE, MIN_PRICE, MIN_PRIOR_RALLY_PCT,
    POINTS, RALLY_WINDOW, STAGE_BANDS, STAGE_ORDER, STAGE_READY, STAGE_SEMI,
    STAGE_WATCH, chart_data, evaluate_event, find_surge_event, technicals,
)

WATCHLIST_FILE = "watchlist.json"
DATA_FILE = "data.json"
STALE_REUSE_DAYS = 3          # إعادة استخدام آخر تقييم إذا فشل تحميل البيانات

# وضع قراءة Float:
#   "auto"   : Float الدقيق من Yahoo، وإذا لم يتوفر يُستخدم عدد الأسهم المُصدَرة كحد أعلى
#              (قبول آمن: Float ≤ أسهم مُصدَرة، فإذا كانت ≤ 10M فالسهم منخفض الـ Float قطعًا).
#   "strict" : Float الدقيق فقط، وأي سهم بلا Float دقيق يُستبعد (أدق لكنه يفرّغ الرادار
#              لأن Yahoo لا يعيد floatShares من GitHub Actions).
FLOAT_MODE = os.environ.get("FLOAT_MODE", "auto")


def universe():
    """رموز الأسهم المتداولة (بدون صناديق/إصدارات تجريبية)."""
    out = set()
    for url, col in [
        ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
        ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol"),
    ]:
        try:
            d = pd.read_csv(url, sep="|")
            d = d[d["Test Issue"] == "N"]
            if "ETF" in d.columns:
                d = d[d["ETF"] != "Y"]
            out.update(str(x).strip().upper() for x in d[col].dropna())
        except Exception as e:
            print("universe error:", e)
    return sorted(x for x in out if x.isalpha() and len(x) <= 5)


def fetch(ticker, interval="1d"):
    """يومي: 6 أشهر · ساعي (1h): 3 أشهر (yfinance لا يدعم 4h مباشرة)."""
    period = "6mo" if interval == "1d" else "3mo"
    d = yf.download(
        ticker, period=period, interval=interval,
        auto_adjust=False, progress=False, threads=False,
    )
    if d is None or d.empty:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    d = d.dropna(subset=needed)
    return d if not d.empty else None


def resample_4h(d):
    """إعادة تجميع شموع 1h إلى 4h (لأن yfinance لا يوفر فريم 4h)."""
    if d is None or d.empty:
        return None
    try:
        x = d.resample("4h").agg({
            "Open": "first", "High": "max", "Low": "min",
            "Close": "last", "Volume": "sum",
        }).dropna(subset=["Open", "High", "Low", "Close"])
        return x if not x.empty else None
    except Exception:
        return None


def get_float_info(ticker, mode=None):
    """Float/حد أعلى من Yahoo ثم SEC EDGAR (انظر float_lookup)."""
    return float_lookup.resolve(ticker, mode=mode or FLOAT_MODE)


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def is_stale_usable(prev, now):
    """هل يمكن إعادة استخدام آخر تقييم إذا فشل تحميل البيانات الآن؟"""
    if not prev or prev.get("has_warning"):
        return False
    try:
        ts = datetime.fromisoformat(str(prev.get("computed_at") or prev.get("updated_at")))
    except Exception:
        return False
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (now - ts) <= timedelta(days=STALE_REUSE_DAYS)


def main():
    ap = argparse.ArgumentParser(description="Bottom Radar scanner")
    ap.add_argument("--tickers", help="رموز محددة للفحص (مفصولة بفاصلة) — للتجربة")
    ap.add_argument("--limit", type=int, help="أقصى عدد رموز للفحص — للتجربة")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true", help="لا تكتب أي ملفات")
    args = ap.parse_args()

    now = datetime.now(timezone.utc)
    symbols = ([s.strip().upper() for s in args.tickers.split(",") if s.strip()]
               if args.tickers else universe())
    if args.limit:
        symbols = symbols[:args.limit]
    if not args.tickers and len(symbols) < 500:
        print("::warning title=Scan skipped::قائمة رموز NASDAQ غير متاحة (عدد الرموز %d) — لم يتم لمس data.json" % len(symbols))
        raise SystemExit(0)
    print("symbols:", len(symbols))

    watch = {x.get("ticker"): x for x in load_json(WATCHLIST_FILE, {}).get("items", []) if x.get("ticker")}
    prev_payload = load_json(DATA_FILE, {})
    prev_signals = {x.get("ticker"): x for x in prev_payload.get("signals", []) if x.get("ticker")}

    # ---------- 1) تحميل يومي + فلاتر سريعة (سعر/صعود/صلاحية زمنية) ----------
    candidates = {}      # ticker -> (daily_df, event)
    checked = 0
    downloaded = 0
    funnel = {"price_pass": 0, "rally_pass": 0, "days_pass": 0}

    def prepass(ticker):
        nonlocal downloaded
        d = fetch(ticker, "1d")
        if d is None or len(d) < RALLY_WINDOW + 25:
            return None
        downloaded += 1
        price = float(d["Close"].iloc[-1])
        if not MIN_PRICE <= price <= MAX_PRICE:
            return None
        funnel["price_pass"] += 1
        event = find_surge_event(d)
        if not event:
            return None
        funnel["rally_pass"] += 1
        if len(d) - 1 - int(event["event_idx"]) > MAX_DAYS_SINCE_PEAK:
            return None
        funnel["days_pass"] += 1
        return ticker, d, event

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(prepass, t): t for t in symbols}
        for fut in as_completed(futures):
            checked += 1
            try:
                res = fut.result()
            except Exception as e:
                print("prepass error", futures[fut], e)
                res = None
            if res:
                ticker, d, event = res
                candidates[ticker] = (d, event)
            if checked % 500 == 0:
                print("scanned", checked, "of", len(symbols), "| candidates:", len(candidates))

    coverage = downloaded / max(1, len(symbols))
    print("daily data coverage: %.1f%% (%d/%d)" % (coverage * 100, downloaded, len(symbols)))

    # إعادة محاولة واحدة عند تقييد Yahoo (Too Many Requests) — لأهم جزء من الرموز فقط
    if coverage < 0.60 and not args.tickers:
        retry_pool = [t for t in symbols if t not in candidates][:2000]
        print("retrying", len(retry_pool), "symbols after temporary Yahoo throttling ...")
        time.sleep(20)
        with ThreadPoolExecutor(max_workers=max(2, args.workers // 2)) as pool:
            futures = {pool.submit(prepass, t): t for t in retry_pool}
            for fut in as_completed(futures):
                try:
                    res = fut.result()
                except Exception:
                    res = None
                if res:
                    ticker, d, event = res
                    candidates[ticker] = (d, event)
                    downloaded += 1
        coverage = downloaded / max(1, len(symbols))
        print("daily data coverage after retry: %.1f%% (%d/%d)" % (coverage * 100, downloaded, len(symbols)))

    if coverage < 0.10:
        print("::warning title=Scan skipped::تغطية بيانات Yahoo ضعيفة (%.1f%%) — لم يتم لمس data.json" % (coverage * 100))
        raise SystemExit(0)

    print("candidates after price/rally/20-day filters:", len(candidates))

    # ---------- 2) Float (شرط صارم) ----------
    floats = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        ff = {pool.submit(get_float_info, t): t for t in candidates}
        for f in as_completed(ff):
            ticker = ff[f]
            try:
                info = f.result()
            except Exception as e:
                print("float error", ticker, e)
                info = None
            if (not info or info.get("value") is None) and watch.get(ticker, {}).get("float_shares"):
                # آخر قيمة معروفة من القائمة المستمرة
                info = {"value": int(watch[ticker]["float_shares"]),
                        "exact": bool(watch[ticker].get("float_exact", True)),
                        "source": watch[ticker].get("float_source") or "watchlist",
                        "float_shares": watch[ticker].get("float_shares"),
                        "shares_outstanding": watch[ticker].get("shares_outstanding")}
            floats[ticker] = info or {"value": None, "exact": False, "source": None,
                                      "float_shares": None, "shares_outstanding": None}
    passed = [t for t, i in floats.items() if i.get("value") is not None and i["value"] <= MAX_FLOAT]
    float_sources = {}
    for i in floats.values():
        if i.get("value") is not None:
            key = i.get("source") or "unknown"
            float_sources[key] = float_sources.get(key, 0) + 1
    funnel["float_available"] = sum(1 for i in floats.values() if i.get("value") is not None)
    funnel["float_exact"] = sum(1 for i in floats.values() if i.get("exact"))
    funnel["float_pass"] = len(passed)
    funnel["float_sources"] = float_sources
    funnel["float_mode"] = FLOAT_MODE
    float_excluded = []
    for t in candidates:
        why = None
        val = (floats.get(t) or {}).get("value")
        if val is None:
            why = "float_missing"
        elif val > MAX_FLOAT:
            why = "float_too_big"
        if not why:
            continue
        print("excluded (%s):" % why, t)
        funnel.setdefault("pre_float_reasons", {})
        funnel["pre_float_reasons"][why] = funnel["pre_float_reasons"].get(why, 0) + 1
        d0, ev0 = candidates[t]
        float_excluded.append({
            "ticker": t, "reason": why,
            "price": round(float(d0["Close"].iloc[-1]), 2),
            "rally_pct": round(float(ev0["rally_pct"]), 1),
            "days_since_peak": int(len(d0) - 1 - int(ev0["event_idx"])),
            "float": val,
        })
        funnel["reasons"] = funnel.get("reasons", {})
        funnel["reasons"][why] = funnel["reasons"].get(why, 0) + 1
    print("candidates after float filter:", len(passed))

    # ---------- 3) شموع 1h/4h والأخبار للمرشحين ----------
    intraday, news_map = {}, {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hf = {pool.submit(fetch, t, "1h"): t for t in passed}
        nf = {pool.submit(fetch_news, t): t for t in passed}
        for f in as_completed(hf):
            t = hf[f]
            try:
                intraday[t] = f.result()
            except Exception as e:
                print("1h error", t, e)
                intraday[t] = None
        for f in as_completed(nf):
            t = nf[f]
            try:
                news_map[t] = f.result()
            except Exception as e:
                print("news error", t, e)
                news_map[t] = {"warnings": [], "catalysts": [], "temp_excluded": False}

    # ---------- 4) التقييم الكامل ----------
    signals, new_watch = [], []
    reject_reasons = dict(funnel.get("reasons", {}))
    near_misses = list(float_excluded)
    for ticker in passed:
        d, event = candidates[ticker]
        h1 = intraday.get(ticker)
        h4 = resample_4h(h1)
        trace = []
        # ملاحظة: لا تستخدم `h1 or h4` مع DataFrame (يُقيّم boolean فيرفع استثناءً)
        intraday_df = h1 if h1 is not None else h4
        try:
            fi = floats[ticker]
            item = evaluate_event(d, event, fi.get("value"), news_map.get(ticker), intraday_df,
                                  now=now, trace=trace, float_exact=bool(fi.get("exact")),
                                  float_source=fi.get("source"),
                                  shares_outstanding=fi.get("shares_outstanding"))
        except Exception as e:
            print("evaluate error", ticker, e)
            item = None
            trace.append("error")
        if item is None:
            reason = trace[-1] if trace else "unknown"
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
            near_misses.append({
                "ticker": ticker, "reason": reason,
                "price": round(float(d["Close"].iloc[-1]), 2),
                "rally_pct": round(float(event["rally_pct"]), 1),
                "days_since_peak": int(len(d) - 1 - int(event["event_idx"])),
                "float": (floats.get(ticker) or {}).get("value"),
            })
            continue
        item["ticker"] = ticker
        item["computed_at"] = now.isoformat()
        if h4 is not None:
            item["technical_4h"] = technicals(h4.tail(30), 30)
        if h1 is not None:
            item["technical_1h"] = technicals(h1.tail(30), 30)
        item["chart"] = chart_data(d, 30)
        meta = watch.get(ticker, {})
        item["discovered_at"] = meta.get("discovered_at") or now.isoformat()
        if item["stage"] == STAGE_READY:
            item["ready_since"] = meta.get("ready_since") or now.isoformat()
        elif meta.get("ready_since"):
            item["ready_since"] = meta["ready_since"]
        signals.append(item)
        new_watch.append({
            "ticker": ticker, "event_date": item["event_date"],
            "discovered_at": item["discovered_at"], "float_shares": item["float_shares"],
            "float_exact": item.get("float_exact"), "float_source": item.get("float_source"),
            "shares_outstanding": item.get("shares_outstanding"),
            "ready_since": item.get("ready_since"),
            "last_seen": now.isoformat(), "last_stage": item["stage"],
            "last_score": item["readiness_score"],
        })

    # إعادة استخدام آخر تقييم إذا فشل تحميل البيانات الآن (يمنع اختفاء السهم من الداشبورد)
    seen = {x["ticker"] for x in signals}
    for ticker in list(candidates) + list(prev_signals):
        if ticker in seen:
            continue
        prev = prev_signals.get(ticker)
        if prev and ticker in candidates and is_stale_usable(prev, now):
            prev = dict(prev)
            prev["stale"] = True
            signals.append(prev)
            print("kept previous evaluation (data fetch failed):", ticker)

    # الأحدث أولًا: جاهز فنيًا ثم شبه جاهز ثم قيد المتابعة ثم الأقدم
    signals.sort(key=lambda x: (
        0 if x.get("has_warning") else 1,
        -STAGE_ORDER.get(x.get("stage"), 0),
        -int(x.get("readiness_score", 0)),
    ))

    near_misses.sort(key=lambda x: (x["days_since_peak"], -x["rally_pct"]))
    diagnostics = {
        "symbols_total": len(symbols),
        "data_coverage_pct": round(coverage * 100, 1),
        "price_pass": funnel.get("price_pass", 0),
        "prior_rally_pass": funnel.get("rally_pass", 0),
        "days_since_peak_pass": funnel.get("days_pass", 0),
        "float_available": funnel.get("float_available", 0),
        "float_exact": funnel.get("float_exact", 0),
        "float_pass": funnel.get("float_pass", 0),
        "float_mode": funnel.get("float_mode", FLOAT_MODE),
        "float_sources": funnel.get("float_sources", {}),
        "reject_reasons": reject_reasons,
        "near_misses": near_misses[:15],
    }

    payload = {
        "updated_at": now.isoformat(),
        "strategy": "Bottom Radar — Second Leg Setup",
        "count": len(signals),
        "diagnostics": diagnostics,
        "params": {
            "price_range": [MIN_PRICE, MAX_PRICE],
            "prior_rally_pct_min": MIN_PRIOR_RALLY_PCT,
            "rally_window_days": RALLY_WINDOW,
            "max_float": MAX_FLOAT,
            "max_days_since_peak": MAX_DAYS_SINCE_PEAK,
            "scoring": POINTS,
            "stage_bands": {k: list(v) for k, v in STAGE_BANDS.items()},
            "stop_buffer_pct": 2,
            "fib_target": 0.382,
            "catalyst_window_days": [7, 30],
            "warning_window_days": 5,
        },
        "stats": {
            "ready": sum(x.get("stage") == STAGE_READY for x in signals),
            "semi_ready": sum(x.get("stage") == STAGE_SEMI for x in signals),
            "watching": sum(x.get("stage") == STAGE_WATCH for x in signals),
            "warnings": sum(bool(x.get("has_warning")) for x in signals),
            "catalysts": sum(bool(x.get("has_upcoming_catalyst")) for x in signals),
        },
        "signals": signals,
    }

    # إذا كانت التغطية ضعيفة (Yahoo متعثر) وصفر نتائج: نحتفظ بالبيانات السابقة بدل مسحها.
    # أما إذا كانت التغطية جيدة فهذا يعني فعليًا «لا يوجد سهم مطابق» ونكتب النتيجة الحقيقية (حتى لو كانت صفرًا).
    if not signals and prev_signals and coverage < 0.5 and not args.tickers:
        print("::warning title=Scan skipped::لا نتائج مع تغطية %.0f%% — تم الاحتفاظ ببيانات المسح السابق" % (coverage * 100))
        raise SystemExit(0)

    if args.dry_run:
        print("dry-run: no files written")
    else:
        save_json(DATA_FILE, payload)
        save_json(WATCHLIST_FILE, {"updated_at": now.isoformat(), "items": new_watch})

    print("signals:", len(signals),
          "| ready:", payload["stats"]["ready"],
          "| semi:", payload["stats"]["semi_ready"],
          "| watching:", payload["stats"]["watching"],
          "| warnings:", payload["stats"]["warnings"])


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        code = e.code if isinstance(e.code, str) else ""
        if code:
            print("::error title=Scanner stopped::" + code.splitlines()[0][:400])
        raise
    except Exception as e:                                  # noqa: BLE001
        import traceback
        print("::error title=Scanner crashed::%s: %s" % (type(e).__name__, str(e)[:300]))
        traceback.print_exc()
        raise
