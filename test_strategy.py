# -*- coding: utf-8 -*-
"""اختبارات الحالات الثمانية الإلزامية (بيانات افتراضية، بدون إنترنت)."""
import numpy as np
import pandas as pd
import strategy as st


def build(closes, vols=None, hl=0.01):
    idx = pd.bdate_range(end="2026-09-22", periods=len(closes))
    c = np.array(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + hl)
    l = np.minimum(o, c) * (1 - hl)
    v = np.array(vols if vols is not None else [1_000_000] * len(c), dtype=float)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)


def seg(a, b, n):
    """مسار هندسي (نسبة ثابتة يوميًا) — أقرب لهبوط الأسهم الفعلي."""
    return list(np.geomspace(a, b, n + 1)[1:])


def base_series(base=2.0, peak=5.0, decline_to=2.05):
    """
    قاعدة أفقية -> صعود سريع إلى القمة -> تآكل بطيء ثم نزيف حاد نحو القاع.
    (النزيف الحاد في آخر أيام الهبوط هو ما يدخل RSI منطقة التشبع البيعي.)
    """
    rng = np.random.default_rng(1)
    flat = list(base * (1 + rng.normal(0, 0.004, 28)))
    up = seg(base, peak, 2)
    slow = seg(peak, peak * 0.96, 15)
    crash = seg(peak * 0.96, decline_to, 4)
    return flat + up + slow + crash


def case1():
    """$2 -> $5 -> $2 و RSI<30  => قيد المتابعة"""
    d = build(base_series())
    r = st.analyze(d)
    assert r is not None, "الحالة 1: يجب أن يدخل الرادار"
    assert r["status"] == st.STAGE_WATCH, r["status"]
    lo, hi = st.STAGE_BANDS[st.STAGE_WATCH]
    assert lo <= r["readiness_score"] <= hi, r["readiness_score"]
    assert r["technical"]["rsi"] < 30 or r["technical"]["rsi_oversold_recent"]
    return r


def case2():
    """نفس السهم: RSI يرتفع من ~27 إلى ~38 ويثبت => شبه جاهز"""
    c = base_series(decline_to=2.0)
    # ثبات ثم ارتداد بسيط مع قاع أعلى
    c += [1.97, 2.0, 2.02, 2.01, 2.03, 2.06, 2.05, 2.08, 2.07, 2.10]
    d = build(c)
    r = st.analyze(d)
    assert r is not None, "الحالة 2: يجب أن يبقى في الرادار"
    assert r["status"] == st.STAGE_SEMI, (r["status"], r["readiness_raw"], r["signal_flags"])
    lo, hi = st.STAGE_BANDS[st.STAGE_SEMI]
    assert lo <= r["readiness_score"] <= hi
    return r


def case3():
    """استعادة EMA20 + VWAP + فوليوم جيد + كسر Lower High قرب القاع => جاهز فنيًا"""
    rng = np.random.default_rng(3)
    flat = list(2.0 * (1 + rng.normal(0, 0.004, 24)))
    c = (flat + seg(2.0, 4.2, 2) + seg(4.2, 2.4, 12) + seg(2.4, 2.2, 12)
         + seg(2.2, 2.25, 1)                         # Lower High عند ~2.25
         + seg(2.25, 1.95, 3)                        # القاع (RSI<30)
         + seg(1.95, 2.05, 1) + seg(2.05, 2.34, 3))  # تعافٍ بفوليوم قوي، ما زال قرب القاع
    vols = [1_000_000] * (len(c) - 4) + [900_000, 2_500_000, 2_800_000, 3_000_000]
    d = build(c, vols)
    r = st.analyze(d)
    assert r is not None, "الحالة 3: يجب أن يبقى في الرادار"
    assert r["status"] == st.STAGE_READY, (r["status"], r["readiness_raw"], r["signal_flags"])
    lo, hi = st.STAGE_BANDS[st.STAGE_READY]
    assert lo <= r["readiness_score"] <= hi
    p = r["plan"]
    assert p["stop"] < p["entry_low"] <= p["entry_high"], p      # الوقف أسفل القاع، الدخول عند القاع
    assert r["support"]["distance_pct"] <= 20, r["support"]
    assert p["targets_detail"] and all(t["gain_pct"] > 0 for t in p["targets_detail"])
    return r


def case4():
    """ظهور Offering => تحذير / استبعاد مؤقت"""
    import news
    now = pd.Timestamp("2026-09-23", tz="UTC").to_pydatetime()
    items = [{"title": "XYZ Announces Pricing of $5 Million Public Offering", "date": "2026-09-22",
              "url": "https://finance.yahoo.com/n/1", "source": "Yahoo Finance"}]
    out = news.analyze_news(items, [], now=now)
    assert out["warnings"], "الحالة 4: يجب ظهور تحذير"
    assert out["temp_excluded"] is True
    return out


def case5():
    """محفز إيجابي مستقبلي => 🚀 محفز قادم"""
    import news
    now = pd.Timestamp("2026-09-23", tz="UTC").to_pydatetime()
    out = news.analyze_news([], [pd.Timestamp("2026-10-05").date()], now=now)
    assert out["catalysts"], "الحالة 5: يجب ظهور محفز"
    assert out["catalysts"][0]["days_until"] == 12
    assert not out["warnings"]
    return out


def case6():
    """$4.20 => خارج الرادار"""
    c = base_series(base=3.0, peak=7.0, decline_to=4.2)
    assert st.analyze(build(c)) is None


def case7():
    """$0.80 => خارج الرادار"""
    c = base_series(base=0.5, peak=1.2, decline_to=0.8)
    assert st.analyze(build(c)) is None


def case8():
    """$3.50 حاليًا وكان $7 => لا يُستبعد بسبب $7"""
    c = base_series(base=3.2, peak=7.0, decline_to=3.5)
    d = build(c)
    r = st.analyze(d)
    assert r is not None, "الحالة 8: لا يجب استبعاده"
    assert r["prior_high"] > 7 * 0.99 and r["price"] <= 4.0


def case9_float():
    """Float > 10M => مستبعد؛ Float غير معروف => لا يُستبعد"""
    d = build(base_series())
    assert st.analyze(d, float_shares=12_000_000) is None
    assert st.analyze(d, float_shares=9_900_000) is not None
    assert st.analyze(d, float_shares=None) is not None


def case11_missed_entry():
    """السهم صعد بعد التعافي (+46% فوق القاع) => فات الدخول، خارج الرادار (مثل PFAI)"""
    c = base_series(decline_to=2.0) + seg(2.0, 2.1, 2) + seg(2.1, 3.0, 6)
    assert st.analyze(build(c)) is None


def case10_real_breakdown():
    """كسر قوي واستمرار القيعان => إلغاء setup"""
    c = base_series(decline_to=2.05) + seg(2.05, 1.6, 4)
    assert st.analyze(build(c)) is None


if __name__ == "__main__":
    tests = [case1, case2, case3, case4, case5, case6, case7, case8, case9_float, case10_real_breakdown, case11_missed_entry]
    failed = 0
    for t in tests:
        try:
            t()
            print("✅", t.__name__, "-", (t.__doc__ or "").strip().splitlines()[0])
        except Exception as e:
            failed += 1
            print("❌", t.__name__, "-", repr(e))
    raise SystemExit(1 if failed else 0)
