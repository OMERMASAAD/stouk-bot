# -*- coding: utf-8 -*-
"""
اختبار شامل للسكانر (بدون إنترنت): نشغّل scanner.main() ببيانات اصطناعية ونفحص
مخرجات data.json/watchlist.json وتشخيص المسح، بما في ذلك مسار استبعاد Float.

هذا الاختبار يحمي من أخطاء التشغيل الحقيقية مثل NameError في مسارات الاستبعاد
أو كتابة نتيجة فارغة فوق بيانات سابقة.
"""
import json
import os
import shutil
import sys
import tempfile

import pandas as pd

import scanner
import strategy as st
from test_strategy import READY_RATES, WATCH_RATES, decline_setup


def _news(now=None):
    return {"warnings": [], "catalysts": [], "temp_excluded": False}


def build_fake(tickers, floats, data=None):
    """يجهّز دوالًا وهمية بدل Yahoo (نداءات الشبكة).

    floats: {ticker: value}  → Float دقيق، أو dict {"value","exact","source",...} للتحكم الكامل.
    """
    frames = data or {}

    def fake_fetch(ticker, interval="1d"):
        if ticker not in frames:
            return None
        daily, hourly = frames[ticker]
        return daily if interval == "1d" else hourly

    def fake_universe():
        return list(tickers)

    def fake_float(ticker, mode=None):
        v = floats.get(ticker)
        if isinstance(v, dict):
            return v
        if v is None:
            # حد أعلى من الأسهم المُصدَرة فقط (كما تفعل SEC) — يُرفض في الوضع الصارم
            return {"value": None, "exact": False, "source": None,
                    "float_shares": None, "shares_outstanding": None}
        return {"value": int(v), "exact": True, "source": "yahoo_float",
                "float_shares": int(v), "shares_outstanding": int(v)}

    def fake_news(ticker, now=None):
        return _news(now)

    return fake_fetch, fake_universe, fake_float, fake_news


def run_scanner(tmp, tickers, floats, data=None, patch_news=None, float_mode="auto"):
    """يشغّل main() على رموز محددة ويكتب في مجلد مؤقت."""
    scanner.DATA_FILE = os.path.join(tmp, "data.json")
    scanner.WATCHLIST_FILE = os.path.join(tmp, "watchlist.json")
    fetch, universe, getf, getn = build_fake(tickers, floats, data)
    scanner.fetch = fetch
    scanner.universe = universe
    scanner.get_float_info = getf
    scanner.fetch_news = patch_news or getn
    old_mode, argv = scanner.FLOAT_MODE, sys.argv
    scanner.FLOAT_MODE = float_mode
    sys.argv = ["scanner.py", "--tickers", ",".join(tickers)]
    try:
        scanner.main()
    finally:
        sys.argv = argv
        scanner.FLOAT_MODE = old_mode
    with open(scanner.DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def case01_ready_stock_end_to_end():
    """سهم مكتمل الشروط: يُكتب في data.json بمرحلة «جاهز فنيًا» مع تشخيص المسح."""
    tmp = tempfile.mkdtemp()
    try:
        d, h1 = decline_setup(READY_RATES, rec_bars=6, rec_rate=0.022)
        payload = run_scanner(tmp, ["VSA"], {"VSA": 4_200_000}, {"VSA": (d, h1)})
        assert payload["count"] == 1, payload["count"]
        sig = payload["signals"][0]
        assert sig["ticker"] == "VSA"
        assert sig["stage"] == st.STAGE_READY and sig["readiness_score"] >= 80
        assert sig["float"] == "4.2M"
        assert sig["plan"]["stop_loss"] < sig["price"] <= sig["plan"]["target_2"]
        assert sig["chart"] and len(sig["chart"]) >= 20
        assert "technical_1h" in sig
        diag = payload["diagnostics"]
        assert diag["symbols_total"] == 1 and diag["float_pass"] == 1
        assert "news" in diag and "float_sources" in diag and "reject_reasons" in diag
        assert diag["data_coverage_pct"] == 100.0
        # watchlist محفوظة
        with open(scanner.WATCHLIST_FILE, encoding="utf-8") as f:
            wl = json.load(f)
        assert wl["items"][0]["ticker"] == "VSA" and wl["items"][0]["float_shares"] == 4_200_000
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def case02_float_paths_no_crash():
    """مسار استبعاد Float (غير متوفر/أكبر من 10M) يعمل ويُسجّل السبب بلا أخطاء تشغيل."""
    tmp = tempfile.mkdtemp()
    try:
        d, h1 = decline_setup(READY_RATES, rec_bars=6, rec_rate=0.022)
        big, big_h1 = decline_setup(READY_RATES, rec_bars=6, rec_rate=0.022, base=2.4)
        payload = run_scanner(
            tmp, ["AAA", "BBB", "CCC"],
            {"AAA": None, "BBB": 25_000_000, "CCC": 3_100_000},
            {"AAA": (d, h1), "BBB": (big, big_h1), "CCC": (d, h1)},
        )
        assert payload["count"] == 1 and payload["signals"][0]["ticker"] == "CCC"
        reasons = payload["diagnostics"]["reject_reasons"]
        assert reasons.get("float_missing") == 1 and reasons.get("float_too_big") == 1, reasons
        misses = {m["ticker"]: m["reason"] for m in payload["diagnostics"]["near_misses"]}
        assert misses.get("AAA") == "float_missing" and misses.get("BBB") == "float_too_big", misses
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def case03_zero_results_writes_diagnostics():
    """صفر نتائج مع تغطية جيدة: يُكتب ملف حقيقي (count=0) مع أسباب الاستبعاد."""
    tmp = tempfile.mkdtemp()
    try:
        d, h1 = decline_setup(READY_RATES, rec_bars=6, rec_rate=0.022)
        payload = run_scanner(tmp, ["ZZZ"], {"ZZZ": 8_000_000}, {"ZZZ": (d, h1)},
                              patch_news=lambda t, now=None: _news())
        # السهم نفسه مقبول؛ لنجبر الرفض بعمر قمة > 20 جلسة
        d2 = pd.concat([d, d.tail(6)])
        payload = run_scanner(tmp, ["ZZZ"], {"ZZZ": 8_000_000}, {"ZZZ": (d2, h1)})
        assert payload["count"] == 0
        diag = payload["diagnostics"]
        assert diag["days_since_peak_pass"] == 0, diag
        assert diag["symbols_total"] == 1 and diag["data_coverage_pct"] == 100.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def case04_watch_and_ready_sorted():
    """الترتيب: جاهز فنيًا أولًا ثم شبه جاهز ثم قيد المتابعة."""
    tmp = tempfile.mkdtemp()
    try:
        # 30 جلسة قاعدة + الصعود + الهبوط = 46 شمعة (السكانر يتجاهل ما دون 45 شمعة)
        ready, ready_h1 = decline_setup(READY_RATES, rec_bars=6, rec_rate=0.022, flat_bars=30)
        watch, watch_h1 = decline_setup(WATCH_RATES, rec_bars=0, rec_rate=0.0, flat_bars=30)
        payload = run_scanner(
            tmp, ["RDY", "WCH"], {"RDY": 2_000_000, "WCH": 5_000_000},
            {"RDY": (ready, ready_h1), "WCH": (watch, watch_h1)},
        )
        order = [(s["ticker"], s["stage"]) for s in payload["signals"]]
        assert payload["count"] == 2, order
        assert order[0][0] == "RDY" and order[0][1] == st.STAGE_READY, order
        assert payload["stats"]["ready"] == 1 and payload["stats"]["watching"] == 1, payload["stats"]
        assert payload["params"]["max_days_since_peak"] == 20
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def case05_float_upper_bound_mode():
    """عند غياب Float الدقيق: يُقبل الحد الأعلى (أسهم مُصدَرة ≤ 10M) في الوضع auto ويُرفض في strict."""
    tmp = tempfile.mkdtemp()
    try:
        d, h1 = decline_setup(READY_RATES, rec_bars=6, rec_rate=0.022)
        proxy = {"value": 6_000_000, "exact": False, "source": "sec_edgar",
                 "float_shares": None, "shares_outstanding": 6_000_000}
        payload = run_scanner(tmp, ["SECY"], {"SECY": proxy}, {"SECY": (d, h1)}, float_mode="auto")
        assert payload["count"] == 1, payload["count"]
        sig = payload["signals"][0]
        assert sig["float"] == "≤ 6.0M" and sig["float_exact"] is False, sig["float"]
        assert sig["float_source"] == "sec_edgar"
        assert payload["diagnostics"]["float_sources"] == {"sec_edgar": 1}
        assert payload["diagnostics"]["float_exact"] == 0

        big = {"value": 40_000_000, "exact": False, "source": "sec_edgar",
               "float_shares": None, "shares_outstanding": 40_000_000}
        payload2 = run_scanner(tmp, ["BIGC"], {"BIGC": big}, {"BIGC": (d, h1)}, float_mode="auto")
        assert payload2["count"] == 0
        assert payload2["diagnostics"]["reject_reasons"].get("float_too_big") == 1

        # الوضع الصارم: لا قبول إلا بـ Float دقيق
        strict_none = {"value": None, "exact": False, "source": None,
                       "float_shares": None, "shares_outstanding": 6_000_000}
        payload3 = run_scanner(tmp, ["STRC"], {"STRC": strict_none}, {"STRC": (d, h1)}, float_mode="strict")
        assert payload3["count"] == 0
        assert payload3["diagnostics"]["reject_reasons"].get("float_missing") == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    tests = [case01_ready_stock_end_to_end, case02_float_paths_no_crash,
             case03_zero_results_writes_diagnostics, case04_watch_and_ready_sorted,
             case05_float_upper_bound_mode]
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
