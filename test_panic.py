# -*- coding: utf-8 -*-
"""اختبار بدون إنترنت لقنص الذعر: بيانات 5m اصطناعية."""
from datetime import datetime, timezone
import numpy as np, pandas as pd
import panic_scanner as ps

def make(post_low_bars=30, crash_to=0.60, flat_noise=0.004, seed=1, start_price=2.0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(end="2026-10-06", periods=4, tz="America/New_York")
    rows = []
    for d in days[:-1]:                                  # أيام سابقة هادئة
        for k in range(120):
            ts = d.replace(hour=9, minute=30) + pd.Timedelta(minutes=5 * k)
            p = start_price * (1 + rng.normal(0, 0.002)); rows.append((ts, p, p*1.003, p*0.997, p, 20000))
    d = days[-1]; ts0 = d.replace(hour=4); p = start_price
    path = [p * (1 + 0.5 * i / 12) for i in range(12)]            # قفزة صباحية
    peak = path[-1]
    path += list(np.linspace(peak, peak * crash_to, 10))           # انهيار عمودي
    low = path[-1]
    for i in range(post_low_bars):                                 # ثبات + تجميع (ميل صاعد خفيف)
        path.append(low * (1 + 0.02 * i / post_low_bars + rng.normal(0, flat_noise)))
    for i, p in enumerate(path):
        hi, lo = p * 1.004, p * 0.996
        up = i > 0 and p >= path[i - 1]
        vol = 400000 if 12 <= i < 22 else (70000 if up else 25000)     # تجميع: حجم أعلى على الصعود
        rows.append((ts0 + pd.Timedelta(minutes=5 * i), p, hi, lo, p, vol))
    df = pd.DataFrame(rows, columns=["ts", "Open", "High", "Low", "Close", "Volume"]).set_index("ts")
    return df, df.index[-1].to_pydatetime().astimezone(timezone.utc)

def t_pass():
    df, now = make()
    res, why = ps.evaluate_panic(df, now)
    assert res and res["drop_pct"] <= -30 and res["hold_min"] >= 60 and abs(res["target"] / res["price"] - 1.25) < 1e-3, (res, why)

def t_short_base():
    df, now = make(post_low_bars=8)
    assert ps.evaluate_panic(df, now)[1] in ("no_base", "indicators"), ps.evaluate_panic(df, now)

def t_no_drop():
    df, now = make(crash_to=0.9)
    assert ps.evaluate_panic(df, now)[1] == "no_drop"

def t_stale():
    df, now = make()
    assert ps.evaluate_panic(df, now + pd.Timedelta(hours=3))[1] == "stale"

def t_purge():
    df, now = make()
    master = {"tickers": [{"ticker": "AAA", "float": 3e6, "float_status": "exact"}], "built_on": "x"}
    out1 = ps.run(master, {}, {"AAA": df}, now)
    assert len(out1["items"]) == 1
    base = out1["items"][0]["base_low"]
    d2 = df.copy(); d2.iloc[-1, d2.columns.get_loc("Close")] = base * 0.96     # كسر -4% < -3%
    out2 = ps.run(master, out1, {"AAA": d2}, now + pd.Timedelta(minutes=15))
    assert not out2["items"] and out2["purged"][0]["ticker"] == "AAA"
    out3 = ps.run(master, out2, {"AAA": df}, now + pd.Timedelta(minutes=30))   # لا يعود في نفس الجلسة
    assert not out3["items"]

if __name__ == "__main__":
    bad = 0
    for f in (t_pass, t_short_base, t_no_drop, t_stale, t_purge):
        try: f(); print("✅", f.__name__)
        except Exception as e: bad += 1; print("❌", f.__name__, repr(e)[:300])
    raise SystemExit(1 if bad else 0)
