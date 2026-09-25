# -*- coding: utf-8 -*-
"""
اختبارات استراتيجية «رادار القاع — الرجل الثانية» على بيانات افتراضية (بدون إنترنت).

كل اختبار يعبر عن قاعدة من المواصفات:
  الفلاتر الصارمة (السعر/الصعود/Float/انتهاء 20 جلسة)، النقاط التسع، المراحل الثلاث،
  خطة الصفقة، التحذيرات (تخفيف/تقسيم عكسي/SEC/شطب/إفلاس/نتائج مخيبة)، والمحفزات.
"""
import numpy as np
import pandas as pd

import news
import strategy as st

NOW = pd.Timestamp("2026-09-23", tz="UTC").to_pydatetime()


# ---------------- أدوات بناء البيانات ----------------
def build(closes, volumes=None, hl=0.004):
    idx = pd.bdate_range(end="2026-09-22", periods=len(closes))
    c = np.array(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) * (1 + hl)
    l = np.minimum(o, c) * (1 - hl)
    v = np.array(volumes if volumes is not None else [1_000_000] * len(c), dtype=float)
    return pd.DataFrame({"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}, index=idx)


def seg(a, b, n):
    """مسار هندسي (نسبة ثابتة يوميًا)."""
    return list(np.geomspace(a, b, n + 1)[1:])


def flat(base, n, jitter=0.003, seed=7):
    rng = np.random.default_rng(seed)
    return list(base * (1 + rng.normal(0, jitter, n)))


def h1_from_daily(d, per_day=4):
    """يبني فريم ساعي (1h) من الشموع اليومية — كما يفعل السكانر مع بيانات Yahoo."""
    rows = []
    for _, row in d.iterrows():
        o, c, h, l = float(row.Open), float(row.Close), float(row.High), float(row.Low)
        path = list(np.geomspace(o, c, per_day + 1))[1:]
        prev = o
        for i, px in enumerate(path):
            hi = h if i == 0 else max(prev, px)
            lo = l if i == per_day - 1 else min(prev, px)
            rows.append({"Open": prev, "High": hi, "Low": lo, "Close": px,
                         "Volume": float(row.Volume) / per_day})
            prev = px
    idx = pd.date_range(end="2026-09-22 15:00", periods=len(rows), freq="h")
    return pd.DataFrame(rows, index=idx)


def decline_setup(rates, rec_bars, rec_rate, flat_bars=26, base=2.0, pump=(3.2, 5.0)):
    """
    قاعدة أفقية ← صعود سريع (+150%) ← هبوط متسارع (تصفية) ← تعافٍ هادئ بفوليوم ضعيف.
    """ 
    closes = flat(base, flat_bars) + list(pump)
    for r in rates:
        closes.append(closes[-1] * (1 - r))
    for _ in range(rec_bars):
        closes.append(closes[-1] * (1 + rec_rate))
    vols = ([1_000_000] * flat_bars + [4_000_000] * len(pump)
            + [3_000_000, 6_000_000, 9_000_000] + [2_000_000] * max(0, len(rates) - 3)
            + [250_000] * rec_bars)
    d = build(closes, vols[:len(closes)])
    return d, h1_from_daily(d)


# سيناريو «جاهز فنيًا»: قاع مزدوج، ثبات، RSI متعافٍ، فوليوم جاف، فوق VWAP، MACD متحسن، محفز قادم
READY_RATES = [0.1075, 0.106, 0.0996, 0.0869, 0.0821, 0.0775, 0.0728, 0.07, 0.0686, 0.065, 0.0568]
# سيناريو «قيد المتابعة»: عاد إلى القاع وما زال يسجل قيعانًا وفوليومه جف (بلا تعافٍ بعد)
WATCH_RATES = [0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.015]


def catalyst_news(days=12):
    date = (NOW.date() + pd.Timedelta(days=days))
    return news.analyze_news([], [date], now=NOW)


def setup_ready():
    """سهم استوفى الفلاتر الصلبة ودرجته >= 80 (جاهز فنيًا) مع محفز قادم."""
    d, h1 = decline_setup(READY_RATES, rec_bars=6, rec_rate=0.022)
    return d, st.analyze(d, float_shares=3_000_000, news=catalyst_news(), intraday=h1, now=NOW)


def setup_watch():
    """سهم عاد إلى القاع وما زال يسجل قيعانًا بفوليوم جاف (بلا تعافٍ/نمط) => قيد المتابعة."""
    d, h1 = decline_setup(WATCH_RATES, rec_bars=0, rec_rate=0.0)
    return d, st.analyze(d, float_shares=3_000_000, intraday=h1, now=NOW)


# ---------------- 1) الفلاتر الصارمة ----------------
def case01_price_filter():
    """فلتر السعر: $6.20 أعلى من الحد و$0.80 أقل من الحد => مستبعد."""
    d, _ = decline_setup(READY_RATES, 6, 0.022)
    for target in (6.2, 0.8):
        x = d.copy()
        scale = target / float(x["Close"].iloc[-1])
        for col in ("Open", "High", "Low", "Close"):
            x[col] = x[col] * scale
        assert st.analyze(x, float_shares=3_000_000) is None, f"سعر {target} يجب أن يُستبعد"


def case02_prior_rally_filter():
    """فلتر الصعود السابق: +60% فقط => مستبعد (المطلوب +100%)."""
    d = build(flat(2.0, 30) + seg(2.0, 3.2, 3) + seg(3.2, 2.05, 8) + [2.02, 2.0, 2.03, 2.01, 2.02, 2.0])
    assert st.find_surge_event(d) is None
    assert st.analyze(d, float_shares=3_000_000) is None


def case03_float_filter():
    """فلتر Float: > 10M مستبعد · غير متوفر مستبعد · = 10M مقبول."""
    d, _ = decline_setup(READY_RATES, 6, 0.022)
    assert st.analyze(d, float_shares=13_000_000) is None
    assert st.analyze(d, float_shares=None) is None, "Float غير المتوفر يجب أن يُستبعد"
    assert st.analyze(d, float_shares=10_000_000, intraday=h1_from_daily(d), now=NOW) is not None


def case04_day_expiration_filter():
    """فلتر انتهاء الصلاحية: قمة عمرها > 20 جلسة => استبعاد فوري."""
    d, h1 = decline_setup(READY_RATES, 6, 0.022)
    r = st.analyze(d, float_shares=3_000_000, intraday=h1, now=NOW)
    assert r is not None and r["days_since_peak"] <= st.MAX_DAYS_SINCE_PEAK, r and r["days_since_peak"]
    longer = build(list(d["Close"]) + [float(d["Close"].iloc[-1])] * 6,
                   list(d["Volume"]) + [200_000] * 6)
    assert st.analyze(longer, float_shares=3_000_000) is None, "بعد 20 جلسة من القمة يُستبعد السهم"


# ---------------- 2) النقاط التسع ----------------
def case05_near_base_points():
    """القرب من القاع: داخل 15% => +20، وأكثر من 20% => 0 نقاط بلا وصف «قريب من القاع»."""
    d, r = setup_ready()
    assert r is not None
    item = [c for c in r["checklist"] if c["key"] == "near_base"][0]
    assert r["distance_from_base_pct"] <= 15.0
    assert item["status"] is True and item["points"] == 20, item

    # تعافٍ ممتد (10 جلسات) => البعد عن القاع أكثر من 20% => صفر نقاط بلا وصف «قريب من القاع»
    far, h1f = decline_setup(READY_RATES, rec_bars=10, rec_rate=0.022)
    rf = st.analyze(far, float_shares=3_000_000, intraday=h1f, now=NOW)
    assert rf is not None and rf["distance_from_base_pct"] > 20.0, rf and rf["distance_from_base_pct"]
    item = [c for c in rf["checklist"] if c["key"] == "near_base"][0]
    assert item["status"] is False and item["points"] == 20, item
    assert item["note"].startswith("البعد الحالي") and "تجاوز 20%" in item["note"], item["note"]


def case06_base_pattern():
    """نمط القاع: قاع مزدوج أو قاع أعلى من قاع => +15 نقطة."""
    d, r = setup_ready()
    item = [c for c in r["checklist"] if c["key"] == "pattern"][0]
    assert item["status"] is True and item["points"] == 15, (item, r["pattern_type"])
    assert r["pattern_type"] in (st.PATTERN_DOUBLE, st.PATTERN_HIGHER_LOW)
    assert r["pattern"]["type"] == r["pattern_type"]

    # اختبار وحدة: قاع مزدوج على فريم ساعي
    base = 2.0
    idx = pd.date_range(end="2026-09-22 15:00", periods=30, freq="h")
    lows = [2.3, 2.25, 2.2, 2.15, 2.1, 2.05, 2.02, 2.06, 2.12, 2.18, 2.2, 2.15, 2.08, 2.03,
            2.02, 2.07, 2.13, 2.19, 2.24, 2.28, 2.3, 2.32, 2.3, 2.28, 2.3, 2.31, 2.32, 2.33, 2.32, 2.34]
    h = pd.DataFrame({"Open": lows, "High": [x * 1.01 for x in lows], "Low": lows,
                      "Close": [x * 1.005 for x in lows], "Volume": [100_000] * 30}, index=idx)
    pat = st.detect_base_pattern(h, base)
    assert pat is not None and pat["type"] == st.PATTERN_DOUBLE, pat

    # اختبار وحدة: قاع أعلى من قاع (القاع الثاني أعلى بأكثر من 5%)
    lows2 = [2.4, 2.3, 2.2, 2.1, 2.0, 1.95, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.1, 2.06,
             2.08, 2.12, 2.18, 2.24, 2.3, 2.36, 2.4, 2.42, 2.4, 2.38, 2.4, 2.41, 2.42, 2.43, 2.42, 2.44]
    h2 = pd.DataFrame({"Open": lows2, "High": [x * 1.01 for x in lows2], "Low": lows2,
                       "Close": [x * 1.005 for x in lows2], "Volume": [100_000] * 30}, index=idx)
    pat2 = st.detect_base_pattern(h2, base)
    assert pat2 is not None and pat2["type"] == st.PATTERN_HIGHER_LOW, pat2


def case07_stable_consolidation():
    """ثبات الدعم لجلستين أو أكثر بلا قاع جديد => +15."""
    d, r = setup_ready()
    item = [c for c in r["checklist"] if c["key"] == "stable"][0]
    assert item["status"] is True and item["points"] == 15, item
    assert r["support"]["holding"] is True
    assert r["support"]["stable_bars"] >= 3


def case08_rsi_turning_up():
    """RSI يتحسن من التشبع البيعي: كان < 30 (يومي أو 1h) والآن > 35 وصاعد => +10."""
    d, r = setup_ready()
    item = [c for c in r["checklist"] if c["key"] == "rsi"][0]
    assert item["status"] is True and item["points"] == 10, (item, r["indicators"])
    assert r["indicators"]["rsi"] > 35 and r["indicators"]["rsi"] > r["indicators"]["rsi_prev"]
    assert (r["technical"]["rsi_min_since_peak"] < 30
            or (r["technical"]["rsi_min_intraday"] or 100) < 30)
    # بلا زيارة لمنطقة التشبع => لا نقاط
    flat_base = build(flat(2.0, 30) + seg(2.0, 5.0, 2) + seg(5.0, 2.2, 14) + flat(2.2, 4, 0.002))
    rf = st.analyze(flat_base, float_shares=3_000_000, now=NOW)
    if rf is not None:
        assert [c for c in rf["checklist"] if c["key"] == "rsi"][0]["status"] is False


def case09_volume_contraction_or_uptick():
    """الفوليوم: جفاف (RVOL <= 0.8) أو زخم خفيف (1.0–1.2) بشمعة خضراء ضيقة فوق EMA9 => +10."""
    d, r = setup_ready()
    item = [c for c in r["checklist"] if c["key"] == "volume"][0]
    assert item["status"] is True and item["points"] == 10, (item, r["indicators"]["rvol"])
    assert r["indicators"]["rvol"] <= 0.8 or 1.0 <= r["indicators"]["rvol"] <= 1.2

    # فوليوم انفجاري (اختراق متأخر) => ليس هو الحالة المطلوبة
    d2 = d.copy()
    d2["Volume"] = list(d2["Volume"])[:-1] + [20_000_000]
    r2 = st.analyze(d2, float_shares=3_000_000, intraday=None, now=NOW)
    if r2 is not None:
        assert r2["indicators"]["rvol"] > 1.2


def case10_ema20_vwap_macd():
    """EMA20 (+10) و VWAP (+5) وتحسن MACD (+5) — شرط التقيّد بالنطاق الطبيعي للسهم."""
    assert st.POINTS["ema20"] == 10 and st.POINTS["vwap"] == 5 and st.POINTS["macd"] == 5
    d, r = setup_ready()
    tech = r["technical"]
    assert r["price"] > tech["vwap"], (r["price"], tech["vwap"])
    assert tech["macd_hist"] > 0 or tech["macd_improving"] is True
    # استعادة EMA20 تتحقق عند التعافي القوي (+18% فوق القاع)، وحينها يخرج السهم من نطاق 15%
    strong, h1 = decline_setup([0.25, 0.20, 0.12, 0.06, 0.04, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02],
                               rec_bars=5, rec_rate=0.03)
    rs = st.analyze(strong, float_shares=3_000_000, intraday=h1, now=NOW)
    assert rs is not None and rs["technical"]["ema20_reclaim"] is True, (rs and rs["technical"])
    assert rs["price"] > rs["technical"]["ema20"]
    assert [c for c in rs["checklist"] if c["key"] == "ema20"][0]["status"] is True
    assert [c for c in rs["checklist"] if c["key"] == "near_base"][0]["status"] is False


def case11_catalyst_points():
    """محفز مستقبلي خلال 7–30 يومًا => +10، وخارج النافذة => 0."""
    d, r = setup_ready()
    item = [c for c in r["checklist"] if c["key"] == "catalyst"][0]
    assert item["status"] is True and item["points"] == 10
    assert r["has_upcoming_catalyst"] is True and "نتائج مالية" in r["catalyst_detail"]

    near = news.analyze_news([], [NOW.date() + pd.Timedelta(days=3)], now=NOW)
    r2 = st.analyze(d, float_shares=3_000_000, news=near, intraday=h1_from_daily(d), now=NOW)
    assert r2 is not None and r2["has_upcoming_catalyst"] is False


# ---------------- 3) المراحل والدرجة وخطة الصفقة ----------------
def case12_stage_bands():
    """شرائح المراحل: 30–50 متابعة · 55–75 شبه جاهز · 80–100 جاهز فنيًا."""
    assert st.stage_from_score(30) == st.STAGE_WATCH
    assert st.stage_from_score(50) == st.STAGE_WATCH
    assert st.stage_from_score(54) == st.STAGE_WATCH
    assert st.stage_from_score(55) == st.STAGE_SEMI
    assert st.stage_from_score(75) == st.STAGE_SEMI
    assert st.stage_from_score(79) == st.STAGE_SEMI
    assert st.stage_from_score(80) == st.STAGE_READY
    assert st.stage_from_score(100) == st.STAGE_READY
    assert st.MAX_DAYS_SINCE_PEAK == 20 and st.MAX_FLOAT == 10_000_000
    assert st.MIN_PRIOR_RALLY_PCT == 100.0 and (st.MIN_PRICE, st.MAX_PRICE) == (1.0, 5.0)


def case13_score_is_sum_of_checklist():
    """الدرجة = مجموع نقاط القائمة (0–100) والمرحلة مشتقة منها."""
    _, r = setup_ready()
    total = sum(c["points"] for c in r["checklist"] if c["status"])
    assert total == r["readiness_score"], (total, r["readiness_score"])
    assert 0 <= r["readiness_score"] <= 100
    assert r["stage"] == st.stage_from_score(r["readiness_score"]) == r["status"]
    assert len(r["checklist"]) == len(st.POINTS) == 9
    assert [c["key"] for c in r["checklist"]] == list(st.POINTS.keys())


def case14_ready_stage_and_plan():
    """«جاهز فنيًا» (>= 80) => خطة صفقة: الدخول = الإغلاق، الوقف = القاع -2%، TP1/TP2."""
    d, r = setup_ready()
    assert r is not None and r["stage"] == st.STAGE_READY, (r and r["stage"], r and r["readiness_score"])
    assert r["readiness_score"] >= 80
    p = r["plan"]
    assert p["entry_price"] == r["price"] == p["entry"]
    assert abs(p["stop_loss"] - r["base_support"] * 0.98) < 0.01 or p["stop_loss"] <= r["base_support"] * 0.99
    assert p["stop_loss"] < r["price"]
    assert p["target_2"] == r["prior_high"], (p["target_2"], r["prior_high"])
    assert p["target_1"] > r["price"] and p["target_2"] >= p["target_1"]
    assert len(p["targets_detail"]) == 2 and all(t["gain_pct"] != 0 for t in p["targets_detail"])
    assert p["ready"] is True


def case15_watch_stage():
    """«قيد المتابعة»: عاد للقاع ويثبت فقط => الدرجة داخل 30–50."""
    d, r = setup_watch()
    assert r is not None, "السهم يجب أن يبقى في الرادار"
    assert r["stage"] == st.STAGE_WATCH, (r["stage"], r["readiness_score"], [(c["key"], c["status"]) for c in r["checklist"]])
    lo, hi = st.STAGE_BANDS[st.STAGE_WATCH]
    assert lo <= r["readiness_score"] <= hi, r["readiness_score"]


def case16_schema_keys_and_arabic_ui():
    """صيغة JSON المطلوبة كاملة (مفاتيح إنجليزية) والواجهة عربية."""
    _, r = setup_ready()
    for k in ["price", "float", "stage", "readiness_score", "days_since_peak", "base_support",
              "distance_from_base_pct", "has_upcoming_catalyst", "catalyst_detail",
              "has_warning", "warning_detail", "pattern_type", "indicators", "checklist"]:
        assert k in r, k
    assert set(["rsi", "ema20", "vwap", "rvol"]).issubset(r["indicators"].keys())
    assert r["float"] == "3.0M"
    assert all(set(["rule", "points", "status"]).issubset(c.keys()) for c in r["checklist"])
    assert r["has_warning"] is False and r["warning_detail"] == ""
    assert r["stage"] in (st.STAGE_WATCH, st.STAGE_SEMI, st.STAGE_READY)
    for c in r["checklist"]:
        assert any("\u0600" <= ch <= "\u06ff" for ch in c["rule"]), c      # نص عربي


# ---------------- 4) التحذيرات والمحفزات ----------------
def _news(items):
    return news.analyze_news(items, [], now=NOW)


def case17_dilution_warning():
    """طرح عام/مباشر/حقوق أولوية/ATM => «🚨 تحذير: خافض لقيمة الأسهم / أزمة»."""
    for title in [
        "XYZ Announces Pricing of $5 Million Public Offering",
        "ABC Announces Registered Direct Offering of Common Stock",
        "DEF Announces Rights Issue to Existing Shareholders",
        "GHI Announces At-The-Market Offering Program",
        "JKL Announces Private Placement Financing",
    ]:
        s = news.summarize_news(_news([{"title": title, "date": "2026-09-22", "url": "", "source": "Yahoo Finance"}]), now=NOW)
        assert s["has_warning"] is True, title
        assert s["warning_detail"].startswith("🚨 تحذير: خافض لقيمة الأسهم / أزمة"), s["warning_detail"]


def case18_severe_events_warning():
    """تقسيم عكسي · تحقيق SEC · إنذار شطب · إفلاس · نتائج مخيبة => تحذيرات."""
    for title in [
        "XYZ Announces Reverse Stock Split",
        "ABC Confirms SEC Investigation Into Accounting",
        "DEF Receives Nasdaq Delisting Notice",
        "GHI Files for Chapter 11 Bankruptcy Protection",
        "JKL Misses Revenue Estimates and Cuts Guidance",
    ]:
        s = news.summarize_news(_news([{"title": title, "date": "2026-09-21", "url": "", "source": "Yahoo Finance"}]), now=NOW)
        assert s["has_warning"] is True, title


def case19_warning_window_and_stock_tag():
    """التحذير لآخر 5 أيام فقط، ويُوسم السهم فيخرج في تبويب «تحذيرات»."""
    old = _news([{"title": "XYZ Announces Pricing of $5 Million Public Offering", "date": "2026-09-10",
                  "url": "", "source": "Yahoo Finance"}])
    assert old["warnings"] == [], "خبر أقدم من 5 أيام لا يُحتسب تحذيرًا"
    d, h1 = decline_setup(READY_RATES, 6, 0.022)
    r = st.analyze(d, float_shares=3_000_000, intraday=h1, now=NOW,
                   news=_news([{"title": "XYZ Announces Pricing of $5 Million Public Offering",
                                "date": "2026-09-22", "url": "", "source": "Yahoo Finance"}]))
    assert r["has_warning"] is True
    assert "🚨" in r["warning_detail"] and "طرح عام" in r["warning_detail"]


def case20_headline_catalyst():
    """عنوان يدل على حدث قادم (FDA/تصويت مساهمين) => محفز مستقبلي."""
    out = _news([{"title": "XYZ to Report Phase 3 Topline Data Next Month", "date": "2026-09-20",
                  "url": "", "source": "Yahoo Finance"}])
    s = news.summarize_news(out, now=NOW)
    assert s["has_upcoming_catalyst"] is True and s["catalyst_detail"].startswith("📅"), s


if __name__ == "__main__":
    tests = [
        case01_price_filter, case02_prior_rally_filter, case03_float_filter, case04_day_expiration_filter,
        case05_near_base_points, case06_base_pattern, case07_stable_consolidation, case08_rsi_turning_up,
        case09_volume_contraction_or_uptick, case10_ema20_vwap_macd, case11_catalyst_points,
        case12_stage_bands, case13_score_is_sum_of_checklist, case14_ready_stage_and_plan,
        case15_watch_stage, case16_schema_keys_and_arabic_ui, case17_dilution_warning,
        case18_severe_events_warning, case19_warning_window_and_stock_tag, case20_headline_catalyst,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print("✅", t.__name__, "-", (t.__doc__ or "").strip().splitlines()[0])
        except Exception as e:
            failed += 1
            print("❌", t.__name__, "-", repr(e))
    print("\n", len(tests) - failed, "/", len(tests), "passed")
    raise SystemExit(1 if failed else 0)
