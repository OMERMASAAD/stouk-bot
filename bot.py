# -*- coding: utf-8 -*-
"""
PULLBACK RECOVERY RADAR
نظام مراقبة الأسهم التي:
1) سعرها الحالي بين 1 و5 دولار.
2) حققت صعوداً سابقاً 100% أو أكثر خلال 4 إلى 20 جلسة.
3) الصعود حديث: قمته خلال آخر 31 يوماً.
4) بعد الصعود عادت للتصحيح قرب منطقة بداية الصعود.
5) ننتظر استقرار الدعم واختباره مرتين على الأقل.
6) التحليل الرئيسي على 4 ساعات مبني من بيانات 1 ساعة.
7) التأكيد النهائي على 15 دقيقة.
8) Telegram لا يرسل أثناء المراقبة؛ يرسل فقط عند اكتمال دخول إيجابي.
9) الدخول على دفعات، والأهداف من امتدادات Fibonacci والمقاومات السابقة.

ملاحظة:
هذا نظام فلترة فني وليس ضماناً للربح أو توصية مالية.
"""

import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# ============================================================
# SETTINGS
# ============================================================

MIN_PRICE = 1.00
MAX_PRICE = 5.00

MIN_RALLY_PERCENT = 100.0
MIN_RALLY_SESSIONS = 4
MAX_RALLY_SESSIONS = 20
MAX_RALLY_AGE_DAYS = 31

DAILY_PERIOD = "180d"
HOUR_PERIOD = "30d"
ENTRY_PERIOD = "10d"

MIN_FLOAT = 5_000_000
MAX_FLOAT = 500_000_000

MAX_SHORT = 50_000

SUPPORT_LOW = 0.90
SUPPORT_HIGH = 1.12
SUPPORT_BREAK = 0.97

MIN_SUPPORT_TESTS = 2
MIN_REBOUND = 0.03

RSI_IMPROVEMENT = 1.5
RSI_MAX = 70.0

ENTRY_VOLUME_RATIO = 1.05

# جعل الشروط قابلة للتحقق وليس مستحيلة
MIN_SCORE = 6

CHUNK_SIZE = 100

POSITIVE_KEYWORDS = [
    "approval", "approved", "contract", "partnership",
    "agreement", "acquisition", "merger", "launch",
    "trial", "phase 3", "phase 2", "fda", "revenue",
    "earnings", "profit", "guidance", "upgrade",
    "order", "deal", "strategic", "positive",
]

alerted = set()


# ============================================================
# HELPERS
# ============================================================

def num(x):
    try:
        if x is None:
            return None
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except Exception:
        return None


def price(x):
    x = num(x)
    return "غير متوفر" if x is None else f"${x:.2f}"


def millions(x):
    x = num(x)
    return "غير متوفر" if x is None else f"{x / 1_000_000:.1f} مليون"


def number(x):
    x = num(x)
    return "غير متوفر" if x is None else f"{x:,.0f}"


def clean_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    if isinstance(d.columns, pd.MultiIndex):
        # yfinance قد يرجع MultiIndex عند طلب سهم واحد/عدة أسهم
        level0 = list(d.columns.get_level_values(0))
        if all(c in level0 for c in ["Open", "High", "Low", "Close", "Volume"]):
            d.columns = d.columns.get_level_values(0)
        else:
            d.columns = d.columns.get_level_values(-1)

    d.columns = [str(c).strip() for c in d.columns]

    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in d.columns for c in needed):
        return pd.DataFrame()

    return d.dropna(subset=needed)


# ============================================================
# TICKERS
# ============================================================

def get_all_tickers():
    out = []

    try:
        x = pd.read_csv(
            "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            sep="|",
        )
        x = x[x["Test Issue"] == "N"]
        out += x["Symbol"].dropna().astype(str).tolist()
    except Exception as e:
        print("خطأ NASDAQ:", e, flush=True)

    try:
        x = pd.read_csv(
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
            sep="|",
        )
        x = x[x["Test Issue"] == "N"]
        out += x["ACT Symbol"].dropna().astype(str).tolist()
    except Exception as e:
        print("خطأ NYSE/AMEX:", e, flush=True)

    clean = set()

    for t in out:
        t = t.strip().upper()
        if t.isalpha() and len(t) <= 5:
            clean.add(t)

    return sorted(clean)


# ============================================================
# INDICATORS
# ============================================================

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    m = ema12 - ema26
    s = m.ewm(span=9, adjust=False).mean()

    return m, s, m - s


def build_4h(hourly):
    d = clean_df(hourly)
    if d.empty:
        return pd.DataFrame()

    d = d.sort_index()

    try:
        out = d.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        return out.dropna(subset=["Open", "High", "Low", "Close"])
    except Exception:
        return pd.DataFrame()


# ============================================================
# RALLY DETECTION
# ============================================================

def detect_recent_rally(df):
    d = clean_df(df)

    if len(d) < 60:
        return None

    today = datetime.now().date()
    best = None

    # نبحث في الجزء الحديث فقط
    start_search = max(1, len(d) - 60)

    for end in range(start_search, len(d)):
        high = num(d.iloc[end]["High"])
        if high is None or high <= 0:
            continue

        high_date = pd.Timestamp(d.index[end]).date()

        if (today - high_date).days > MAX_RALLY_AGE_DAYS:
            continue

        for sessions in range(MIN_RALLY_SESSIONS, MAX_RALLY_SESSIONS + 1):
            start = end - sessions

            if start < 0:
                continue

            low = num(d.iloc[start:end + 1]["Low"])
            if low is None or low <= 0:
                continue

            # نستخدم أقل سعر ضمن بداية الحركة
            base = num(d.iloc[start]["Low"])
            if base is None or base <= 0:
                base = low

            rally_pct = ((high - base) / base) * 100

            if rally_pct < MIN_RALLY_PERCENT:
                continue

            # يجب أن تكون القمة بعد بداية الصعود فعلياً
            if high <= base:
                continue

            candidate = {
                "start_pos": start,
                "end_pos": end,
                "start_date": pd.Timestamp(d.index[start]).date(),
                "high_date": high_date,
                "base_price": base,
                "high_price": high,
                "rally_percent": rally_pct,
                "sessions": sessions,
                "rally_volume": num(
                    d.iloc[start:end + 1]["Volume"].mean()
                ),
            }

            if best is None:
                best = candidate
            elif candidate["high_date"] > best["high_date"]:
                best = candidate
            elif (
                candidate["high_date"] == best["high_date"]
                and candidate["rally_percent"] > best["rally_percent"]
            ):
                best = candidate

    return best


# ============================================================
# SUPPORT
# ============================================================

def calculate_support(df, rally):
    if rally is None:
        return None

    d = clean_df(df)
    if d.empty:
        return None

    start = rally["start_pos"]

    # قاع البداية + عدة جلسات قبلها
    begin = max(0, start - 5)
    end = min(len(d), start + 3)

    window = d.iloc[begin:end]
    if window.empty:
        return None

    # دعم البداية يعتمد على القاع الأقرب للحركة
    lows = window["Low"].dropna()
    if lows.empty:
        return None

    return num(lows.min())


def analyze_support_retests(df, support, rally):
    result = {
        "tests": 0,
        "successful_tests": 0,
        "stable": False,
        "second_test": False,
        "last_test": None,
    }

    if support is None or rally is None:
        return result

    d = clean_df(df)
    if d.empty:
        return result

    start_date = rally["start_date"]

    after = d[d.index.date >= start_date].copy()
    if after.empty:
        return result

    lower = support * SUPPORT_LOW
    upper = support * SUPPORT_HIGH

    tests = []

    for idx, row in after.iterrows():
        low = num(row["Low"])
        close = num(row["Close"])

        if low is None:
            continue

        # هل لمس منطقة الدعم؟
        if not (lower <= low <= upper):
            continue

        test_date = pd.Timestamp(idx).date()

        rebound = False
        if close is not None:
            rebound = close >= low * (1 + MIN_REBOUND)

        if not tests:
            tests.append({
                "date": test_date,
                "low": low,
                "rebound": rebound,
            })
        else:
            gap = (test_date - tests[-1]["date"]).days

            if gap >= 2:
                tests.append({
                    "date": test_date,
                    "low": low,
                    "rebound": rebound,
                })
            elif rebound:
                tests[-1]["rebound"] = True

    result["tests"] = len(tests)
    result["successful_tests"] = sum(1 for x in tests if x["rebound"])

    if tests:
        result["last_test"] = tests[-1]["date"]

    result["second_test"] = len(tests) >= MIN_SUPPORT_TESTS
    result["stable"] = result["successful_tests"] >= 1

    return result


# ============================================================
# 4H ANALYSIS
# ============================================================

def analyze_4h(df):
    result = {
        "rsi": None,
        "prev_rsi": None,
        "rsi_improving": False,
        "macd": None,
        "macd_signal": None,
        "macd_positive": False,
        "macd_improving": False,
        "macd_cross": False,
        "volume_ratio": 0,
        "ma20": None,
        "ma50": None,
        "above_ma20": False,
        "above_ma50": False,
    }

    d = clean_df(df)

    if len(d) < 55:
        return result

    close = d["Close"]

    d["RSI"] = calculate_rsi(close)
    d["MACD"], d["MACD_SIGNAL"], d["MACD_HIST"] = calculate_macd(close)
    d["MA20"] = close.rolling(20).mean()
    d["MA50"] = close.rolling(50).mean()

    latest = d.iloc[-1]

    r = num(latest["RSI"])
    pr = num(d["RSI"].iloc[-2])

    m = num(latest["MACD"])
    s = num(latest["MACD_SIGNAL"])

    hist = num(latest["MACD_HIST"])
    prev_hist = num(d["MACD_HIST"].iloc[-2])

    volume = num(latest["Volume"])
    avg_volume = num(d["Volume"].tail(20).mean())

    ma20 = num(latest["MA20"])
    ma50 = num(latest["MA50"])

    result.update({
        "rsi": r,
        "prev_rsi": pr,
        "rsi_improving": (
            r is not None and
            pr is not None and
            r > pr and
            (r - pr) >= RSI_IMPROVEMENT
        ),
        "macd": m,
        "macd_signal": s,
        "macd_positive": (
            m is not None and
            s is not None and
            m > s
        ),
        "macd_improving": (
            hist is not None and
            prev_hist is not None and
            hist > prev_hist
        ),
        "ma20": ma20,
        "ma50": ma50,
        "above_ma20": (
            ma20 is not None and
            close.iloc[-1] >= ma20
        ),
        "above_ma50": (
            ma50 is not None and
            close.iloc[-1] >= ma50
        ),
    })

    if len(d) >= 3:
        m1 = num(d["MACD"].iloc[-1])
        s1 = num(d["MACD_SIGNAL"].iloc[-1])
        m2 = num(d["MACD"].iloc[-2])
        s2 = num(d["MACD_SIGNAL"].iloc[-2])

        result["macd_cross"] = (
            m1 is not None and
            s1 is not None and
            m2 is not None and
            s2 is not None and
            m1 > s1 and
            m2 <= s2
        )

    if volume is not None and avg_volume and avg_volume > 0:
        result["volume_ratio"] = volume / avg_volume

    return result


# ============================================================
# 15M CONFIRMATION
# ============================================================

def confirm_15m(ticker):
    result = {
        "confirmed": False,
        "price": None,
        "rsi": None,
        "rsi_improving": False,
        "macd_positive": False,
        "volume_confirmed": False,
    }

    try:
        df = yf.download(
            ticker,
            period=ENTRY_PERIOD,
            interval="15m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        d = clean_df(df)

        if len(d) < 40:
            return result

        close = d["Close"]

        d["RSI"] = calculate_rsi(close)
        d["MACD"], d["MACD_SIGNAL"], d["MACD_HIST"] = calculate_macd(close)

        latest = d.iloc[-1]

        p = num(latest["Close"])
        r = num(latest["RSI"])
        pr = num(d["RSI"].iloc[-2])

        m = num(latest["MACD"])
        s = num(latest["MACD_SIGNAL"])

        vol = num(latest["Volume"])
        avg_vol = num(d["Volume"].tail(20).mean())

        result["price"] = p
        result["rsi"] = r

        result["rsi_improving"] = (
            r is not None and
            pr is not None and
            r > pr
        )

        result["macd_positive"] = (
            m is not None and
            s is not None and
            m > s
        )

        result["volume_confirmed"] = (
            vol is not None and
            avg_vol is not None and
            avg_vol > 0 and
            vol >= avg_vol * ENTRY_VOLUME_RATIO
        )

        confirmations = sum([
            result["rsi_improving"],
            result["macd_positive"],
            result["volume_confirmed"],
        ])

        # نحتاج اثنين من ثلاثة فقط
        result["confirmed"] = confirmations >= 2

    except Exception:
        pass

    return result


# ============================================================
# FLOAT / SHORT
# ============================================================

def get_float_short(ticker):
    result = {
        "float": None,
        "short": None,
    }

    try:
        info = yf.Ticker(ticker).info

        result["float"] = num(info.get("floatShares"))
        result["short"] = num(info.get("sharesShort"))

    except Exception:
        pass

    return result


# ============================================================
# NEWS
# ============================================================

def get_positive_news(ticker):
    found = []

    try:
        news = yf.Ticker(ticker).news or []

        for item in news[:10]:
            title = ""

            content = item.get("content", {})

            if isinstance(content, dict):
                title = content.get("title") or ""

            if not title:
                title = item.get("title") or ""

            title = str(title)

            if any(k.lower() in title.lower() for k in POSITIVE_KEYWORDS):
                found.append(title)

    except Exception:
        pass

    return found[:3]


# ============================================================
# RESISTANCES / FIB
# ============================================================

def fibonacci_targets(low, high):
    low = num(low)
    high = num(high)

    if low is None or high is None or high <= low:
        return []

    diff = high - low

    return [
        high + diff * 0.272,
        high + diff * 0.618,
        high + diff * 1.000,
        high + diff * 1.618,
    ]


def find_resistances(df, current_price, rally_high):
    d = clean_df(df)

    if d.empty:
        return []

    current_price = num(current_price)
    rally_high = num(rally_high)

    if current_price is None:
        return []

    levels = []

    highs = d["High"].dropna().tail(80)

    for h in highs:
        h = num(h)

        if h is None:
            continue

        if h <= current_price * 1.03:
            continue

        if h > current_price * 5:
            continue

        if not any(abs(h - x) / x < 0.04 for x in levels):
            levels.append(h)

    if rally_high is not None and rally_high > current_price * 1.03:
        if not any(abs(rally_high - x) / x < 0.04 for x in levels):
            levels.append(rally_high)

    return sorted(levels)


def build_targets(current_price, rally_low, rally_high, resistances):
    targets = []

    for x in fibonacci_targets(rally_low, rally_high):
        x = num(x)
        if x is not None and x > current_price * 1.05:
            targets.append(x)

    for x in resistances:
        x = num(x)
        if x is not None and x > current_price * 1.05:
            targets.append(x)

    unique = []

    for x in sorted(targets):
        if not any(abs(x - y) / y < 0.04 for y in unique):
            unique.append(x)

    return unique[:5]


# ============================================================
# STOCK ANALYSIS
# ============================================================

def analyze_stock(ticker):
    try:
        daily = yf.download(
            ticker,
            period=DAILY_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        d = clean_df(daily)

        if len(d) < 60:
            return None

        current_price = num(d["Close"].iloc[-1])

        if current_price is None:
            return None

        # شرط السعر الحالي
        if not (MIN_PRICE <= current_price <= MAX_PRICE):
            return None

        rally = detect_recent_rally(d)

        if rally is None:
            return None

        support = calculate_support(d, rally)

        if support is None or support <= 0:
            return None

        # يجب أن يكون السعر قريباً من دعم بداية الصعود
        if not (
            support * SUPPORT_LOW
            <= current_price
            <= support * SUPPORT_HIGH * 1.35
        ):
            return None

        # لا نريد سهماً كسر الدعم بشكل واضح
        if current_price < support * SUPPORT_BREAK:
            return None

        support_info = analyze_support_retests(
            d,
            support,
            rally
        )

        # لا نطلب الاختبارين في هذه المرحلة بشكل قاتل.
        # يكفي اختبار واحد ناجح أو اختبارين إجمالاً.
        support_ready = (
            support_info["successful_tests"] >= 1
            or support_info["tests"] >= MIN_SUPPORT_TESTS
        )

        if not support_ready:
            return None

        # بيانات 1H -> 4H
        hourly = yf.download(
            ticker,
            period=HOUR_PERIOD,
            interval="1h",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        h = clean_df(hourly)
        four_h = build_4h(h)

        if len(four_h) < 55:
            return None

        tech = analyze_4h(four_h)

        # تأكيد 15 دقيقة
        small = confirm_15m(ticker)

        # Float / Short
        fs = get_float_short(ticker)

        float_shares = fs["float"]
        short_shares = fs["short"]

        # float غير المتوفر لا يمنع الفرصة
        if float_shares is not None:
            if float_shares < MIN_FLOAT:
                return None

            if float_shares > MAX_FLOAT:
                return None

        # Short عامل إيجابي فقط، ليس شرطاً
        short_positive = (
            short_shares is not None
            and short_shares < MAX_SHORT
        )

        # أخبار
        catalysts = get_positive_news(ticker)
        catalyst_positive = len(catalysts) > 0

        # ====================================================
        # SCORE
        # ====================================================

        score = 0
        reasons = []

        # 1 صعود قوي
        score += 2
        reasons.append(
            f"صعود سابق +{rally['rally_percent']:.0f}%"
        )

        # 2 قرب الدعم
        score += 2
        reasons.append("السعر قرب دعم بداية الصعود")

        # 3 استقرار الدعم
        if support_info["successful_tests"] >= 1:
            score += 2
            reasons.append("اختبار دعم ناجح")

        if support_info["tests"] >= MIN_SUPPORT_TESTS:
            score += 1
            reasons.append("وجود اختبارين للدعم")

        # 4 RSI
        if tech["rsi"] is not None:
            if tech["rsi"] <= RSI_MAX:
                score += 1
                reasons.append("RSI غير متضخم")

            if tech["rsi_improving"]:
                score += 2
                reasons.append("RSI يتحسن")

        # 5 MACD
        if tech["macd_positive"]:
            score += 2
            reasons.append("MACD إيجابي")

        if tech["macd_improving"]:
            score += 1
            reasons.append("MACD يتحسن")

        if tech["macd_cross"]:
            score += 2
            reasons.append("تقاطع MACD إيجابي")

        # 6 متوسطات
        if tech["above_ma20"]:
            score += 1
            reasons.append("السعر فوق MA20")

        if tech["above_ma50"]:
            score += 1
            reasons.append("السعر فوق MA50")

        # 7 الفريم الصغير
        if small["confirmed"]:
            score += 2
            reasons.append("تأكيد فريم 15 دقيقة")

        # 8 حجم
        if small["volume_confirmed"]:
            score += 1
            reasons.append("حجم دخول أفضل من المتوسط")

        # 9 short
        if short_positive:
            score += 1
            reasons.append("Short أقل من 50 ألف")

        # 10 خبر
        if catalyst_positive:
            score += 2
            reasons.append("محفز إيجابي")

        # ====================================================
        # دخول نهائي
        # ====================================================

        if score < MIN_SCORE:
            return None

        # يجب أن يكون التأكيد الفني موجوداً
        technical_ok = (
            tech["rsi_improving"]
            or tech["macd_positive"]
            or tech["macd_improving"]
            or small["confirmed"]
        )

        if not technical_ok:
            return None

        # منطقة الدخول حول الدعم، مع توسيع بسيط
        entry_low = support * 1.00
        entry_high = support * 1.30

        if current_price < entry_low:
            entry_low = current_price * 0.98

        if current_price > entry_high:
            entry_high = current_price

        stop_loss = support * 0.94

        resistances = find_resistances(
            d,
            current_price,
            rally["high_price"]
        )

        targets = build_targets(
            current_price,
            rally["base_price"],
            rally["high_price"],
            resistances
        )

        # إذا لم نجد أهدافاً، نستخدم مقاومة بسيطة من الحركة
        if not targets:
            targets = [
                current_price * 1.10,
                current_price * 1.20,
                current_price * 1.35,
            ]

        return {
            "ticker": ticker,
            "price": current_price,

            "rally_percent": rally["rally_percent"],
            "rally_sessions": rally["sessions"],
            "rally_low": rally["base_price"],
            "rally_high": rally["high_price"],
            "rally_start": str(rally["start_date"]),
            "rally_high_date": str(rally["high_date"]),

            "support": support,
            "entry_low": entry_low,
            "entry_high": entry_high,
            "stop_loss": stop_loss,

            "targets": targets,
            "resistances": resistances[:5],

            "score": score,
            "reasons": reasons,

            "rsi": tech["rsi"] or 0,
            "rsi_improving": tech["rsi_improving"],
            "macd_positive": tech["macd_positive"],
            "macd_improving": tech["macd_improving"],
            "macd_cross": tech["macd_cross"],
            "volume_ratio": tech["volume_ratio"],

            "support_tests": support_info["tests"],
            "successful_tests": support_info["successful_tests"],

            "float_shares": float_shares,
            "short_shares": short_shares,

            "short_positive": short_positive,
            "catalysts": catalysts,

            "small_confirmed": small["confirmed"],
            "small_rsi": small["rsi"],
        }

    except Exception as e:
        print(f"[{ticker}] خطأ:", e, flush=True)
        return None


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(result):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print(
            "⚠️ لم يتم إرسال Telegram: الأسرار غير موجودة.",
            flush=True
        )
        return False

    targets = result["targets"]

    target_lines = []
    for i, target in enumerate(targets, 1):
        target_lines.append(
            f"🎯 الهدف {i}: {price(target)}"
        )

    targets_text = "\n".join(target_lines)

    if result["catalysts"]:
        news_text = (
            "📰 محفزات إيجابية:\n"
            + "\n".join(
                f"• {x}"
                for x in result["catalysts"]
            )
        )
    else:
        news_text = "📰 لا يوجد محفز إخباري واضح حالياً"

    message = (
        "🚨 إشارة دخول إيجابية 🚨\n\n"
        f"📌 السهم: {result['ticker']}\n"
        f"💵 السعر الحالي: {price(result['price'])}\n\n"

        "📈 الحركة السابقة:\n"
        f"• الصعود: +{result['rally_percent']:.1f}%\n"
        f"• مدة الصعود: {result['rally_sessions']} جلسات\n"
        f"• قاع الحركة: {price(result['rally_low'])}\n"
        f"• قمة الحركة: {price(result['rally_high'])}\n\n"

        "🟢 منطقة الدخول على دفعات:\n"
        f"• {price(result['entry_low'])} → {price(result['entry_high'])}\n\n"

        f"🟡 الدعم: {price(result['support'])}\n"
        f"⛔ وقف الخسارة: {price(result['stop_loss'])}\n\n"

        f"{targets_text}\n\n"

        "📊 التأكيدات:\n"
        f"• التقييم: {result['score']}\n"
        f"• RSI: {result['rsi']:.1f}\n"
        f"• RSI يتحسن: {'نعم' if result['rsi_improving'] else 'لا'}\n"
        f"• MACD إيجابي: {'نعم' if result['macd_positive'] else 'لا'}\n"
        f"• MACD يتحسن: {'نعم' if result['macd_improving'] else 'لا'}\n"
        f"• تقاطع MACD: {'نعم' if result['macd_cross'] else 'لا'}\n"
        f"• حجم التداول: {result['volume_ratio']:.2f}x\n"
        f"• اختبارات الدعم: {result['support_tests']}\n"
        f"• اختبارات ناجحة: {result['successful_tests']}\n"
        f"• تأكيد 15 دقيقة: {'نعم' if result['small_confirmed'] else 'لا'}\n\n"

        f"💎 الفلوت: {millions(result['float_shares'])}\n"
        f"📉 Short: {number(result['short_shares'])}\n\n"

        f"{news_text}\n\n"

        "🧠 أهم أسباب الإشارة:\n"
        + "\n".join(
            f"• {x}"
            for x in result["reasons"]
        )
        + "\n\n"

        "📌 طريقة التعامل:\n"
        "الدخول تدريجي وعلى دفعات، وليس بكامل السيولة من أول سعر.\n"
        "الأهداف موزعة على المقاومات وامتدادات فيبوناتشي.\n\n"

        "⚠️ تنبيه آلي فني وليس توصية مالية. "
        "راجع الخبر والشارت وإدارة المخاطر قبل الدخول."
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    try:
        r = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
            },
            timeout=20,
        )

        if r.ok:
            print(
                f"✅ Telegram: تم إرسال {result['ticker']}",
                flush=True
            )
            return True

        print("❌ Telegram:", r.text, flush=True)
        return False

    except Exception as e:
        print("❌ خطأ Telegram:", e, flush=True)
        return False


# ============================================================
# DASHBOARD / MANUAL RESULT
# ============================================================

def print_result(result):
    print("\n" + "=" * 65)
    print(f"🚨 مرشح إيجابي: {result['ticker']}")
    print("=" * 65)

    print(f"السعر الحالي       : {price(result['price'])}")
    print(f"الصعود السابق      : +{result['rally_percent']:.1f}%")
    print(f"مدة الصعود         : {result['rally_sessions']} جلسات")
    print(f"قاع الصعود         : {price(result['rally_low'])}")
    print(f"قمة الصعود         : {price(result['rally_high'])}")
    print(f"الدعم              : {price(result['support'])}")
    print(
        f"منطقة الدخول       : "
        f"{price(result['entry_low'])} - {price(result['entry_high'])}"
    )
    print(f"وقف الخسارة        : {price(result['stop_loss'])}")

    print("\nالأهداف:")
    for i, x in enumerate(result["targets"], 1):
        print(f"  الهدف {i}: {price(x)}")

    print("\nالتأكيدات:")
    print(f"  Score              : {result['score']}")
    print(f"  RSI                : {result['rsi']:.1f}")
    print(f"  RSI يتحسن          : {result['rsi_improving']}")
    print(f"  MACD إيجابي        : {result['macd_positive']}")
    print(f"  MACD يتحسن         : {result['macd_improving']}")
    print(f"  MACD Cross          : {result['macd_cross']}")
    print(f"  Volume Ratio        : {result['volume_ratio']:.2f}x")
    print(f"  اختبارات الدعم     : {result['support_tests']}")
    print(f"  اختبارات ناجحة     : {result['successful_tests']}")
    print(f"  تأكيد 15m          : {result['small_confirmed']}")

    print("\nالفلوت / الشورت:")
    print(f"  Float              : {millions(result['float_shares'])}")
    print(f"  Short              : {number(result['short_shares'])}")

    print("\nالأسباب:")
    for x in result["reasons"]:
        print(f"  • {x}")

    print("\nالأخبار:")
    if result["catalysts"]:
        for x in result["catalysts"]:
            print(f"  • {x}")
    else:
        print("  لا يوجد خبر إيجابي واضح")

    print("=" * 65 + "\n")


# ============================================================
# SCAN
# ============================================================

def scan_all():
    print("\n" + "=" * 65)
    print("🔎 بدء فحص PULLBACK RECOVERY RADAR")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 65, flush=True)

    tickers = get_all_tickers()

    if not tickers:
        print("❌ لم يتم تحميل قائمة الأسهم.", flush=True)
        return

    print(
        f"تم تحميل {len(tickers)} سهم.",
        flush=True
    )

    candidates = []

    total = (len(tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        n = i // CHUNK_SIZE + 1

        print(
            f"\n[دفعة {n}/{total}] "
            f"فحص {len(chunk)} سهم...",
            flush=True
        )

        for ticker in chunk:
            result = analyze_stock(ticker)

            if result is None:
                continue

            candidates.append(result)

            print_result(result)

            # Telegram فقط عند الإشارة النهائية
            key = (
                ticker,
                result["rally_high_date"],
            )

            if key not in alerted:
                if send_telegram(result):
                    alerted.add(key)

    print("\n" + "=" * 65)
    print(
        f"✅ انتهى الفحص. عدد الإشارات الإيجابية: {len(candidates)}"
    )
    print("=" * 65, flush=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # الفحص اليدوي أو فحص GitHub Actions
    scan_all()
