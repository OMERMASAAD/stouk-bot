# -*- coding: utf-8 -*-
"""
⚡ قنص الذعر اللحظي (Panic Dip-Buy) — يعمل كل 15 دقيقة على رموز master_low_float.json فقط.

الشروط (كلها معًا): هبوط ≤ -30% من قمة اليوم · ثبات أفقي (تذبذب ≤ 5%) لمدة ≥ 60 دقيقة قرب قاع اليوم ·
RSI(5m) يخرج من التشبع البيعي · OBV صاعد أو انحراف إيجابي · MACD تقاطع/هيستوجرام أخضر.
التنظيف الآلي: كسر قاع الثبات بـ -3% إضافية => يُشطب السهم ولا يعود في نفس الجلسة.
المخرجات: panic_data.json
"""
import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from strategy import ema, macd, rsi

MASTER_FILE, OUT_FILE = "master_low_float.json", "panic_data.json"
ET = ZoneInfo("America/New_York")

DROP_MAX_PCT = -30.0          # هبوط من قمة اليوم
CONS_RANGE_PCT = 5.0          # أقصى تذبذب للثبات الأفقي
CONS_MIN_MIN = 60             # أقل زمن ثبات (دقائق)
NEAR_LOW_PCT = 5.0            # قاع نافذة الثبات ضمن 5% من أدنى قاع اليوم
RSI_OVERSOLD, RSI_EXIT, RSI_LOOK = 30.0, 25.0, 24     # آخر 24 شمعة = ساعتان
OBV_LOOK = 12
PURGE_BREAK_PCT = 3.0         # كسر قاع الثبات => شطب
TARGET_PCT = 25.0
STALE_MIN = 60                # آخر شمعة أقدم من ساعة => السهم متوقف/ميت
CHUNK = 100


def obv(close, volume):
    return (np.sign(close.diff().fillna(0)) * volume).cumsum()


def strength_score(checks, near_low):
    """درجة تحليلية من 100، وليست توصية شراء."""
    weights = {"base": 35, "rsi": 20, "obv": 20, "macd": 15}
    score = sum(weight for key, weight in weights.items() if checks.get(key))
    score += 10 if near_low else 0
    label = "مكتمل وقوي" if score >= 80 else ("قريب من الاكتمال" if score >= 55 else "يحتاج متابعة")
    return score, label


def chart_points(df, limit=72):
    """نقاط آخر 6 ساعات تقريبًا للرسم اللحظي في الداشبورد."""
    x = df.tail(limit)
    return [{
        "time": idx.isoformat(),
        "open": round(float(row["Open"]), 4),
        "high": round(float(row["High"]), 4),
        "low": round(float(row["Low"]), 4),
        "close": round(float(row["Close"]), 4),
        "volume": int(row["Volume"]),
    } for idx, row in x.iterrows()]


def _et_index(df):
    idx = df.index
    df = df.copy()
    df.index = idx.tz_localize("UTC").tz_convert(ET) if idx.tz is None else idx.tz_convert(ET)
    return df


def evaluate_panic(df, now=None):
    """
    يعيد (dict | None, reason). أي سهم هبط ≤ -30% من قمة اليوم يُعاد مع حالة كل شرط في "checks"،
    و"complete"=True فقط عند تحقق كل الشروط (ثبات + RSI + OBV + MACD).
    """
    if df is None or len(df) < 30:
        return None, "no_data"
    df = _et_index(df)
    now = (now or datetime.now(timezone.utc)).astimezone(ET)
    last_ts = df.index[-1]
    if (now - last_ts).total_seconds() / 60 > STALE_MIN:
        return None, "stale"
    day = df[df.index.date == last_ts.date()]
    if len(day) < 12:
        return None, "no_data"
    price = float(day["Close"].iloc[-1])
    day_high, day_low = float(day["High"].max()), float(day["Low"].min())
    drop = (price / day_high - 1) * 100
    if drop > DROP_MAX_PCT:
        return None, "no_drop"
    high_ts = day["High"].idxmax()
    after = day.loc[high_ts:]
    # --- الثبات الأفقي: نمشي للخلف طالما المدى ≤ 5% ---
    hi, lo, start_ts = -1e18, 1e18, last_ts
    for ts, row in after.iloc[::-1].iterrows():
        hi, lo = max(hi, float(row["High"])), min(lo, float(row["Low"]))
        if (hi - lo) / lo * 100 > CONS_RANGE_PCT:
            break
        start_ts = ts
    win = after.loc[start_ts:]
    base_low = float(win["Low"].min())
    hold_min = int((last_ts - start_ts).total_seconds() / 60) + 5
    near_low = (base_low - day_low) / day_low * 100 <= NEAR_LOW_PCT
    base_ok = hold_min >= CONS_MIN_MIN and near_low
    # --- المؤشرات على كامل الـ 5 أيام (تسخين كافٍ) ---
    close, vol = df["Close"].astype(float), df["Volume"].astype(float)
    r = rsi(close, 14)
    rsi_now = rsi_min = None
    rsi_ok = False
    if r.dropna().shape[0] >= RSI_LOOK:
        rsi_now, rsi_min = float(r.iloc[-1]), float(r.iloc[-RSI_LOOK:].min())
        rsi_ok = rsi_min <= RSI_OVERSOLD and rsi_now >= RSI_EXIT and rsi_now - rsi_min >= 3
    o = obv(close, vol)
    lows = df["Low"].astype(float)
    obv_rising = bool(o.iloc[-1] > o.iloc[-OBV_LOOK])
    obv_div = bool(lows.iloc[-OBV_LOOK:].min() <= lows.iloc[-2 * OBV_LOOK:-OBV_LOOK].min()
                   and o.iloc[-OBV_LOOK:].min() > o.iloc[-2 * OBV_LOOK:-OBV_LOOK].min())
    obv_ok = obv_rising or obv_div
    ml, ms, mh = macd(close)
    cross = bool(((ml.shift(1) <= ms.shift(1)) & (ml > ms)).iloc[-3:].any())
    macd_ok = cross or float(mh.iloc[-1]) > 0
    checks = {"base": bool(base_ok), "rsi": bool(rsi_ok), "obv": bool(obv_ok), "macd": bool(macd_ok)}
    score, score_label = strength_score(checks, near_low)
    volume_last = float(vol.iloc[-1])
    volume_avg = float(vol.tail(20).mean())
    rvol = volume_last / volume_avg if volume_avg > 0 else None
    return {
        "price": round(price, 4), "day_high": round(day_high, 4), "day_low": round(day_low, 4),
        "drop_pct": round(drop, 1), "hold_min": hold_min, "hold_needed": CONS_MIN_MIN,
        "base_low": round(base_low, 4), "near_low": bool(near_low),
        "dist_from_low_pct": round((price / day_low - 1) * 100, 1),
        "target": round(price * (1 + TARGET_PCT / 100), 4), "target_pct": TARGET_PCT,
        "rsi": round(rsi_now, 1) if rsi_now is not None else None,
        "rsi_min": round(rsi_min, 1) if rsi_min is not None else None,
        "obv": "صاعد" if obv_rising else ("انحراف إيجابي" if obv_div else "ضعيف"),
        "macd": "تقاطع" if cross else ("هيستوجرام أخضر" if macd_ok else "سلبي"),
        "checks": checks, "complete": all(checks.values()),
        "strength_score": score, "strength_label": score_label,
        "volume_last": int(volume_last), "volume_avg": int(volume_avg),
        "rvol": round(rvol, 2) if rvol is not None else None,
        "dollar_volume": round(price * volume_last, 2),
        "last_bar": last_ts.isoformat(),
    }, "ok"


def download(tickers):
    """تحميل دفعي (100 رمز/طلب) — أخف بكثير على Yahoo من طلب لكل سهم."""
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        for attempt in (1, 2):
            try:
                raw = yf.download(chunk, period="5d", interval="5m", prepost=True, group_by="ticker",
                                  auto_adjust=False, threads=True, progress=False)
            except Exception as e:
                print("download error", type(e).__name__, str(e)[:100]); raw = None
            if raw is not None and not raw.empty:
                break
            time.sleep(5)
        if raw is None or raw.empty:
            continue
        lvl0 = raw.columns.get_level_values(0) if isinstance(raw.columns, pd.MultiIndex) else []
        for t in chunk:
            x = raw[t] if len(lvl0) and t in lvl0 else (raw if len(chunk) == 1 else None)
            if x is not None:
                x = x.dropna(subset=["Open", "High", "Low", "Close"])
                if len(x):
                    out[t] = x
    return out


def load(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def run(master, prev, frames, now):
    """الدمج + التنظيف الآلي. أي هابط ≤ -30% يظهر؛ والشطب لمن ثبت فعلًا ثم كسر قاعه بـ -3%."""
    today = now.astimezone(ET).date().isoformat()
    if prev.get("session_date") != today:
        prev = {}
    items = {x["ticker"]: x for x in prev.get("items", [])}
    purged = {x["ticker"]: x for x in prev.get("purged", [])}
    meta = {m["ticker"]: m for m in master.get("tickers", [])}
    reasons = {}
    for t, m in meta.items():
        df = frames.get(t)
        if df is None:
            reasons["no_data"] = reasons.get("no_data", 0) + 1
            continue
        px = float(df["Close"].iloc[-1])
        old = items.get(t)
        if old and old.get("had_base") and px < old["base_low"] * (1 - PURGE_BREAK_PCT / 100):   # 🧹 تنظيف آلي
            purged[t] = {"ticker": t, "price": round(px, 4), "base_low": old["base_low"],
                         "purged_at": now.isoformat(), "reason": "كسر قاع الثبات بـ -%g%%" % PURGE_BREAK_PCT}
            items.pop(t)
            continue
        if t in purged:
            continue
        res, why = evaluate_panic(df, now)
        reasons[why] = reasons.get(why, 0) + 1
        if res:
            old = old or {}
            had_base = bool(old.get("had_base") or res["checks"]["base"])
            res.update(ticker=t, float=m.get("float"), float_status=m.get("float_status"),
                       detected_at=old.get("detected_at", now.isoformat()), still_valid=True,
                       had_base=had_base,
                       base_low=old["base_low"] if old.get("had_base") else res["base_low"])  # قاع الثبات الأول هو مرجع الشطب
            history = list(old.get("history") or [])
            snapshot = {k: res.get(k) for k in (
                "last_bar", "price", "drop_pct", "hold_min", "strength_score",
                "complete", "checks", "rvol")}
            if not history or history[-1].get("last_bar") != snapshot["last_bar"]:
                history.append(snapshot)
            res["history"] = history[-48:]
            res["chart"] = chart_points(df)
            items[t] = res
        elif t in items:
            if items[t].get("had_base"):                # كان ثابتًا ولم يُكسر قاعه بعد: يبقى مع علامة «ضعفت»
                items[t]["still_valid"] = False
                items[t]["complete"] = False
                items[t]["price"] = round(px, 4)
            else:                                       # ارتد فوق -30% قبل أن يثبت: لم يعد مرشحًا
                items.pop(t)
    ordered = sorted(items.values(), key=lambda x: (not x.get("complete", False), x["drop_pct"]))
    return {"updated_at": now.isoformat(), "session_date": today,
            "master_count": len(meta), "master_built_on": master.get("built_on"),
            "items": ordered, "purged": list(purged.values()), "diagnostics": {"reasons": reasons}}


def main():
    t0, now = time.time(), datetime.now(timezone.utc)
    master = load(MASTER_FILE, {})
    if not master.get("tickers"):
        print("::warning title=No master::master_low_float.json غير موجود — شغّل Stock Radar أولًا")
        return
    frames = download([m["ticker"] for m in master["tickers"]])
    cov = len(frames) / max(1, len(master["tickers"]))
    print("5m coverage: %.0f%% (%d/%d)" % (cov * 100, len(frames), len(master["tickers"])))
    if cov < 0.3:
        print("::warning title=Panic scan skipped::تغطية بيانات 5m ضعيفة — لم يتم لمس panic_data.json")
        return
    out = run(master, load(OUT_FILE, {}), frames, now)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("panic items: %d (complete %d) | purged: %d | %.0fs" % (len(out["items"]), sum(1 for x in out["items"] if x.get("complete")), len(out["purged"]), time.time() - t0))


if __name__ == "__main__":
    main()
