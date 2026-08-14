# -*- coding: utf-8 -*-
"""
PULLBACK RECOVERY RADAR
نظام مراقبة الأسهم التي صعدت بقوة ثم صححت إلى دعم بداية الصعود.

الفكرة:
- السعر الحالي بين 1 و5 دولار.
- صعود سابق لا يقل عن 100% خلال 4 إلى 20 جلسة.
- الصعود حديث: قمته خلال آخر 31 يومًا.
- لا يرسل أي تنبيه أثناء المراقبة.
- يرسل Telegram فقط عندما تكتمل إشارة دخول إيجابية.
- التحليل الرئيسي على 4 ساعات (يُبنى من 1H).
- تأكيد الدخول على 15 دقيقة.
- RSI + MACD + الحجم + المتوسطات + Fibonacci + المقاومات.
- البيع على المكشوف أقل من 50 ألف عامل إيجابي فقط، وليس شرطًا إجباريًا.
- الأخبار الإيجابية عامل مساعد فقط.
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
# الإعدادات
# ============================================================

MIN_PRICE = 1.00
MAX_PRICE = 5.00

MIN_RALLY_PERCENT = 100.0
MIN_RALLY_SESSIONS = 4
MAX_RALLY_SESSIONS = 20

# قمة الصعود يجب أن تكون حديثة، وبحد أدنى جلسة سابقة
MIN_RALLY_AGE_DAYS = 1
MAX_RALLY_AGE_DAYS = 31

# الحد الأدنى للبيانات اليومية
DAILY_PERIOD = "180d"

# 4H يُبنى من 1H
INTRADAY_PERIOD = "30d"
INTRADAY_INTERVAL = "1h"

# فريم الدخول
ENTRY_PERIOD = "10d"
ENTRY_INTERVAL = "15m"

# Float: عامل فلترة وليس شرطًا شديدًا
MIN_FLOAT_SHARES = 5_000_000
MAX_FLOAT_SHARES = 500_000_000

# Short أقل من 50 ألف = عامل إيجابي
MAX_SHORT_SHARES = 50_000

# الدعم
SUPPORT_LOW_MULT = 0.90
SUPPORT_HIGH_MULT = 1.12
SUPPORT_BREAK_MULT = 0.97

# نريد اختبارين للدعم على الأقل
MIN_SUPPORT_TESTS = 2

# الارتداد من الدعم
MIN_REBOUND_PERCENT = 3.0

# مؤشرات 4H
RSI_MIN = 30.0
RSI_MAX = 70.0
RSI_IMPROVEMENT = 1.5

# حجم الارتداد
ENTRY_VOLUME_RATIO = 1.05

# عدد النقاط المطلوبة لإشارة الدخول
MIN_ENTRY_SCORE = 6

# الفحص كل ساعة
SCAN_INTERVAL_SECONDS = 60 * 60

# عدد الأسهم في كل دفعة yfinance
CHUNK_SIZE = 100

# ============================================================
# الذاكرة
# ============================================================

alerted_entries = set()


# ============================================================
# أدوات
# ============================================================

def safe_float(value):
    try:
        if value is None:
            return None
        value = float(value)
        if np.isnan(value) or np.isinf(value):
            return None
        return value
    except Exception:
        return None


def fmt_price(value):
    value = safe_float(value)
    return "غير متوفر" if value is None else f"${value:.2f}"


def fmt_number(value):
    value = safe_float(value)
    return "غير متوفر" if value is None else f"{value:,.0f}"


def fmt_millions(value):
    value = safe_float(value)
    return "غير متوفر" if value is None else f"{value / 1_000_000:.1f} مليون"


def normalize_columns(df):
    """تحويل أعمدة yfinance إلى أعمدة عادية."""
    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    if isinstance(data.columns, pd.MultiIndex):
        # إذا كان مستوى أول هو اسم الحقل
        if all(x in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
               for x in data.columns.get_level_values(0).unique()):
            data.columns = data.columns.get_level_values(0)
        else:
            data.columns = data.columns.get_level_values(-1)

    data.columns = [str(c).strip() for c in data.columns]

    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in data.columns for c in required):
        return pd.DataFrame()

    return data.dropna(subset=required)


# ============================================================
# قائمة الأسهم
# ============================================================

def get_all_tickers():
    tickers = []

    try:
        nasdaq = pd.read_csv(
            "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            sep="|",
        )
        nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]
        tickers.extend(nasdaq["Symbol"].dropna().astype(str).tolist())
    except Exception as exc:
        print(f"خطأ تحميل قائمة NASDAQ: {exc}", flush=True)

    try:
        other = pd.read_csv(
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
            sep="|",
        )
        other = other[other["Test Issue"] == "N"]
        tickers.extend(other["ACT Symbol"].dropna().astype(str).tolist())
    except Exception as exc:
        print(f"خطأ تحميل قائمة NYSE/AMEX: {exc}", flush=True)

    clean = []
    for ticker in tickers:
        ticker = ticker.strip().upper()
        if ticker.isalpha() and len(ticker) <= 5:
            clean.append(ticker)

    return sorted(set(clean))


# ============================================================
# مؤشرات
# ============================================================

def calculate_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal

    return macd, signal, hist


# ============================================================
# اكتشاف أحدث صعود 100%+
# ============================================================

def detect_recent_rally(df):
    if df is None or len(df) < 60:
        return None

    data = normalize_columns(df)
    if data.empty:
        return None

    today = datetime.now().date()
    best = None

    # نبحث عن قمم حديثة، وليس عن أي صعود قديم.
    first_search = max(1, len(data) - 50)

    for end_pos in range(first_search, len(data)):
        high = safe_float(data.iloc[end_pos]["High"])
        if high is None or high <= 0:
            continue

        high_date = pd.Timestamp(data.index[end_pos]).date()
        age_days = (today - high_date).days

        if age_days < MIN_RALLY_AGE_DAYS:
            continue
        if age_days > MAX_RALLY_AGE_DAYS:
            continue

        for sessions in range(
            MIN_RALLY_SESSIONS,
            MAX_RALLY_SESSIONS + 1,
        ):
            start_pos = end_pos - sessions
            if start_pos < 0:
                continue

            start_low = safe_float(data.iloc[start_pos]["Low"])
            if start_low is None or start_low <= 0:
                continue

            # قاع بداية الحركة
            base_price = start_low

            rally_percent = ((high - base_price) / base_price) * 100
            if rally_percent < MIN_RALLY_PERCENT:
                continue

            # تأكد أن الصعود فعلي وليس مجرد شمعة شاذة واحدة:
            # نريد إغلاقًا في منتصف/نهاية الحركة أعلى من البداية.
            start_close = safe_float(data.iloc[start_pos]["Close"])
            end_close = safe_float(data.iloc[end_pos]["Close"])

            if start_close is None or end_close is None:
                continue

            # لا نشترط 100% بالإغلاق؛ الهاي يكفي، لكن الإغلاق يجب ألا يكون
            # انهيارًا كبيرًا تحت منتصف الحركة.
            midpoint = base_price + (high - base_price) * 0.50
            if end_close < midpoint * 0.70:
                continue

            candidate = {
                "start_pos": start_pos,
                "end_pos": end_pos,
                "start_date": pd.Timestamp(data.index[start_pos]).date(),
                "high_date": high_date,
                "base_price": base_price,
                "high_price": high,
                "rally_percent": rally_percent,
                "sessions": sessions,
            }

            # نأخذ أحدث قمة، وإذا تساوت نأخذ الصعود الأقوى.
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
# تحديد دعم بداية الصعود
# ============================================================

def calculate_support(df, rally):
    if rally is None:
        return None

    data = normalize_columns(df)
    if data.empty:
        return None

    start_pos = rally["start_pos"]

    # نأخذ القاع في 5 جلسات قبل بداية الصعود + يوم البداية.
    begin = max(0, start_pos - 5)
    end = min(len(data), start_pos + 2)

    window = data.iloc[begin:end]
    if window.empty:
        return None

    lows = window["Low"].dropna()
    if lows.empty:
        return None

    # القاع الأقرب لبداية الحركة أفضل من قاع بعيد جدًا.
    support = safe_float(lows.min())
    return support


def price_near_support(price, support):
    price = safe_float(price)
    support = safe_float(support)

    if price is None or support is None or support <= 0:
        return False

    return (
        support * SUPPORT_LOW_MULT
        <= price
        <= support * SUPPORT_HIGH_MULT
    )


# ============================================================
# تحليل اختبارات الدعم
# ============================================================

def analyze_support_retests(df, rally, support):
    result = {
        "tests": 0,
        "successful_tests": 0,
        "support_stable": False,
        "second_test": False,
        "last_test_date": None,
    }

    if df is None or rally is None or support is None:
        return result

    data = normalize_columns(df)
    if data.empty:
        return result

    start_date = rally["start_date"]
    after = data[data.index.date >= start_date].copy()

    lower = support * SUPPORT_LOW_MULT
    upper = support * SUPPORT_HIGH_MULT

    tests = []

    for idx, row in after.iterrows():
        low = safe_float(row["Low"])
        close = safe_float(row["Close"])

        if low is None or close is None:
            continue

        if not (lower <= low <= upper):
            continue

        date = pd.Timestamp(idx).date()
        rebound = close >= low * (1 + MIN_REBOUND_PERCENT / 100)

        # لا نحسب عدة شموع متتالية كاختبارات منفصلة.
        if not tests:
            tests.append({
                "date": date,
                "low": low,
                "rebound": rebound,
            })
            continue

        gap = (date - tests[-1]["date"]).days

        if gap >= 2:
            tests.append({
                "date": date,
                "low": low,
                "rebound": rebound,
            })
        elif rebound:
            tests[-1]["rebound"] = True

    result["tests"] = len(tests)
    result["successful_tests"] = sum(1 for x in tests if x["rebound"])

    if tests:
        result["last_test_date"] = tests[-1]["date"]

    result["second_test"] = result["tests"] >= MIN_SUPPORT_TESTS
    result["support_stable"] = result["successful_tests"] >= 1

    return result


def support_not_broken(df, rally, support):
    if support is None or rally is None:
        return False

    data = normalize_columns(df)
    if data.empty:
        return False

    start_pos = rally["start_pos"]
    after = data.iloc[start_pos:]

    break_level = support * SUPPORT_BREAK_MULT

    # إذا حصل إغلاق قوي أسفل الدعم، الفرصة تعتبر فاشلة.
    closes = after["Close"].dropna()
    if closes.empty:
        return False

    return not (closes < break_level).any()


# ============================================================
# بناء 4H من 1H
# ============================================================

def build_4h(df_1h):
    data = normalize_columns(df_1h)
    if data.empty:
        return pd.DataFrame()

    data = data.sort_index()

    try:
        result = data.resample("4h").agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })

        return result.dropna(
            subset=["Open", "High", "Low", "Close"]
        )
    except Exception:
        return pd.DataFrame()


# ============================================================
# تحليل 4H
# ============================================================

def analyze_4h(df_4h):
    result = {
        "ready": False,
        "rsi": None,
        "previous_rsi": None,
        "rsi_improving": False,
        "macd": None,
        "macd_signal": None,
        "macd_hist": None,
        "previous_hist": None,
        "macd_positive": False,
        "macd_improving": False,
        "macd_cross": False,
        "volume_ratio": None,
        "volume_confirmed": False,
        "ma20": None,
        "ma50": None,
        "above_ma20": False,
        "above_ma50": False,
    }

    if df_4h is None or len(df_4h) < 55:
        return result

    data = normalize_columns(df_4h)
    if len(data) < 55:
        return result

    close = data["Close"]

    data["RSI"] = calculate_rsi(close)
    data["MACD"], data["MACD_SIGNAL"], data["MACD_HIST"] = calculate_macd(close)
    data["MA20"] = close.rolling(20).mean()
    data["MA50"] = close.rolling(50).mean()

    latest = data.iloc[-1]
    previous = data.iloc[-2]

    rsi = safe_float(latest["RSI"])
    prev_rsi = safe_float(previous["RSI"])

    macd = safe_float(latest["MACD"])
    signal = safe_float(latest["MACD_SIGNAL"])
    hist = safe_float(latest["MACD_HIST"])
    prev_hist = safe_float(previous["MACD_HIST"])

    volume = safe_float(latest["Volume"])
    avg_volume = safe_float(data["Volume"].tail(20).mean())

    ma20 = safe_float(latest["MA20"])
    ma50 = safe_float(latest["MA50"])

    result.update({
        "ready": True,
        "rsi": rsi,
        "previous_rsi": prev_rsi,
        "rsi_improving": (
            rsi is not None
            and prev_rsi is not None
            and rsi > prev_rsi
            and (rsi - prev_rsi) >= RSI_IMPROVEMENT
        ),
        "macd": macd,
        "macd_signal": signal,
        "macd_hist": hist,
        "previous_hist": prev_hist,
        "macd_positive": (
            macd is not None
            and signal is not None
            and macd > signal
        ),
        "macd_improving": (
            hist is not None
            and prev_hist is not None
            and hist > prev_hist
        ),
        "volume_ratio": (
            volume / avg_volume
            if volume is not None and avg_volume and avg_volume > 0
            else None
        ),
        "volume_confirmed": (
            volume is not None
            and avg_volume is not None
            and avg_volume > 0
            and volume >= avg_volume * ENTRY_VOLUME_RATIO
        ),
        "ma20": ma20,
        "ma50": ma50,
        "above_ma20": ma20 is not None and close.iloc[-1] >= ma20,
        "above_ma50": ma50 is not None and close.iloc[-1] >= ma50,
    })

    # تقاطع MACD صاعد
    prev_macd = safe_float(previous["MACD"])
    prev_signal = safe_float(previous["MACD_SIGNAL"])

    result["macd_cross"] = (
        macd is not None
        and signal is not None
        and prev_macd is not None
        and prev_signal is not None
        and macd > signal
        and prev_macd <= prev_signal
    )

    return result


# ============================================================
# فريم 15 دقيقة
# ============================================================

def confirm_15m(ticker):
    result = {
        "confirmed": False,
        "price": None,
        "rsi": None,
        "rsi_improving": False,
        "macd_positive": False,
        "volume_confirmed": False,
        "score": 0,
    }

    try:
        df = yf.download(
            ticker,
            period=ENTRY_PERIOD,
            interval=ENTRY_INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        data = normalize_columns(df)
        if len(data) < 50:
            return result

        close = data["Close"]

        data["RSI"] = calculate_rsi(close)
        data["MACD"], data["MACD_SIGNAL"], data["MACD_HIST"] = calculate_macd(close)

        latest = data.iloc[-1]
        previous = data.iloc[-2]

        price = safe_float(latest["Close"])
        rsi = safe_float(latest["RSI"])
        prev_rsi = safe_float(previous["RSI"])

        macd = safe_float(latest["MACD"])
        signal = safe_float(latest["MACD_SIGNAL"])

        volume = safe_float(latest["Volume"])
        avg_volume = safe_float(data["Volume"].tail(20).mean())

        result["price"] = price
        result["rsi"] = rsi

        if (
            rsi is not None
            and prev_rsi is not None
            and rsi > prev_rsi
        ):
            result["rsi_improving"] = True
            result["score"] += 1

        if (
            macd is not None
            and signal is not None
            and macd > signal
        ):
            result["macd_positive"] = True
            result["score"] += 1

        if (
            volume is not None
            and avg_volume is not None
            and avg_volume > 0
            and volume >= avg_volume * 1.05
        ):
            result["volume_confirmed"] = True
            result["score"] += 1

        # نحتاج تأكيدين من ثلاثة.
        result["confirmed"] = result["score"] >= 2

    except Exception as exc:
        print(f"[{ticker}] خطأ 15m: {exc}", flush=True)

    return result


# ============================================================
# المقاومات
# ============================================================

def find_resistances(df, current_price, rally_high):
    data = normalize_columns(df)
    if data.empty:
        return []

    current_price = safe_float(current_price)
    rally_high = safe_float(rally_high)

    if current_price is None:
        return []

    levels = []

    # قمم آخر 90 يومًا
    for value in data["High"].dropna().tail(90):
        value = safe_float(value)

        if value is None:
            continue
        if value <= current_price * 1.03:
            continue
        if value > current_price * 4.0:
            continue

        if not any(
            abs(value - x) / x < 0.035
            for x in levels
        ):
            levels.append(value)

    if rally_high is not None and rally_high > current_price * 1.03:
        if not any(
            abs(rally_high - x) / x < 0.035
            for x in levels
        ):
            levels.append(rally_high)

    return sorted(levels)


# ============================================================
# Fibonacci
# ============================================================

def fibonacci_retracements(low, high):
    low = safe_float(low)
    high = safe_float(high)

    if low is None or high is None or high <= low:
        return {}

    diff = high - low

    return {
        "38.2%": high - diff * 0.382,
        "50.0%": high - diff * 0.500,
        "61.8%": high - diff * 0.618,
        "78.6%": high - diff * 0.786,
    }


def fibonacci_targets(low, high):
    low = safe_float(low)
    high = safe_float(high)

    if low is None or high is None or high <= low:
        return []

    diff = high - low

    return [
        high + diff * 0.272,
        high + diff * 0.618,
        high + diff * 1.000,
        high + diff * 1.618,
    ]


# ============================================================
# Float / Short
# ============================================================

def get_float_short(ticker):
    result = {
        "float": None,
        "short": None,
    }

    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        result["float"] = safe_float(info.get("floatShares"))
        result["short"] = safe_float(info.get("sharesShort"))
    except Exception as exc:
        print(f"[{ticker}] تعذر جلب Float/Short: {exc}", flush=True)

    return result


# ============================================================
# أخبار إيجابية
# ============================================================

POSITIVE_KEYWORDS = [
    "approval",
    "approved",
    "contract",
    "partnership",
    "agreement",
    "acquisition",
    "merger",
    "launch",
    "trial",
    "phase 3",
    "phase 2",
    "fda",
    "revenue",
    "earnings",
    "profit",
    "guidance",
    "upgrade",
    "order",
    "deal",
    "strategic",
    "positive",
]


def get_positive_catalysts(ticker):
    catalysts = []

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

            if any(
                keyword in title.lower()
                for keyword in POSITIVE_KEYWORDS
            ):
                catalysts.append(title)

    except Exception:
        pass

    return catalysts[:3]


# ============================================================
# الأهداف + الدخول
# ============================================================

def build_targets(current_price, rally_low, rally_high, resistances):
    targets = []

    for target in fibonacci_targets(rally_low, rally_high):
        target = safe_float(target)
        if target is not None and target > current_price * 1.05:
            targets.append(target)

    for resistance in resistances:
        resistance = safe_float(resistance)
        if resistance is not None and resistance > current_price * 1.05:
            targets.append(resistance)

    unique = []

    for target in sorted(targets):
        if not any(abs(target - x) / x < 0.04 for x in unique):
            unique.append(target)

    return unique[:5]


# ============================================================
# إرسال Telegram بالعربي فقط عند الدخول
# ============================================================

def send_telegram_entry(result):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ مفاتيح Telegram غير موجودة؛ لن يتم إرسال التنبيه.", flush=True)
        return False

    targets_text = "\n".join(
        f"🎯 الهدف {i}: {fmt_price(target)}"
        for i, target in enumerate(result["targets"], 1)
    )

    if not targets_text:
        targets_text = "🎯 الأهداف: لم يتم تحديد أهداف واضحة."

    if result["catalysts"]:
        catalyst_text = (
            "📰 محفزات إيجابية:\n"
            + "\n".join(f"• {x}" for x in result["catalysts"])
        )
    else:
        catalyst_text = "📰 المحفزات: لا يوجد محفز إيجابي واضح في البيانات المتاحة."

    short_positive = (
        result["short_shares"] is not None
        and result["short_shares"] < MAX_SHORT_SHARES
    )

    message = (
        "🚨 إشارة دخول إيجابية 🚨\n\n"
        f"📌 السهم: {result['ticker']}\n"
        f"💵 السعر الحالي: {fmt_price(result['price'])}\n\n"

        "📈 حركة الصعود السابقة:\n"
        f"• الصعود: +{result['rally_percent']:.1f}%\n"
        f"• مدة الصعود: {result['rally_sessions']} جلسات\n"
        f"• بداية الصعود: {fmt_price(result['rally_low'])}\n"
        f"• قمة الصعود: {fmt_price(result['rally_high'])}\n\n"

        "🟢 منطقة الدخول على دفعات:\n"
        f"• من {fmt_price(result['entry_low'])} "
        f"إلى {fmt_price(result['entry_high'])}\n"
        f"• الدعم: {fmt_price(result['support'])}\n"
        f"• وقف الخسارة: {fmt_price(result['stop_loss'])}\n\n"

        f"{targets_text}\n\n"

        "📊 تأكيدات فنية:\n"
        f"• RSI على 4 ساعات: {result['rsi']:.1f}\n"
        f"• RSI يتحسن: {'نعم' if result['rsi_improving'] else 'لا'}\n"
        f"• MACD إيجابي: {'نعم' if result['macd_positive'] else 'لا'}\n"
        f"• MACD يتحسن: {'نعم' if result['macd_improving'] else 'لا'}\n"
        f"• تقاطع MACD: {'نعم' if result['macd_cross'] else 'لا'}\n"
        f"• فوق MA20: {'نعم' if result['above_ma20'] else 'لا'}\n"
        f"• فوق MA50: {'نعم' if result['above_ma50'] else 'لا'}\n"
        f"• نسبة حجم 4 ساعات: "
        f"{result['volume_ratio']:.2f}x\n"
        f"• اختبارات الدعم: {result['support_tests']}\n"
        f"• اختبارات ناجحة: {result['successful_tests']}\n"
        f"• تأكيد 15 دقيقة: {'نعم' if result['small_confirmed'] else 'لا'}\n\n"

        "📌 عوامل إضافية:\n"
        f"• الفلوت: {fmt_millions(result['float_shares'])}\n"
        f"• البيع على المكشوف: {fmt_number(result['short_shares'])}\n"
        f"• البيع على المكشوف أقل من 50 ألف: "
        f"{'نعم' if short_positive else 'لا'}\n\n"

        f"{catalyst_text}\n\n"

        "📍 طريقة التعامل:\n"
        "دخول تدريجي على دفعات بعد ثبات الدعم وإعادة الاختبار، "
        "وليس دخول كامل السيولة من أول سعر.\n\n"

        "⚠️ هذا تنبيه آلي فني وليس توصية مالية. "
        "راجع الخبر والشارت وإدارة المخاطر قبل اتخاذ القرار."
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"

    try:
        response = requests.post(
            url,
            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
            },
            timeout=20,
        )

        if response.ok:
            print(
                f"✅ تم إرسال تنبيه الدخول للسهم {result['ticker']}",
                flush=True,
            )
            return True

        print(
            f"❌ Telegram رفض الرسالة: {response.text}",
            flush=True,
        )
        return False

    except Exception as exc:
        print(f"❌ خطأ Telegram: {exc}", flush=True)
        return False


# ============================================================
# تحليل سهم واحد
# ============================================================

def analyze_stock(ticker):
    try:
        daily_raw = yf.download(
            ticker,
            period=DAILY_PERIOD,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        daily = normalize_columns(daily_raw)

        if len(daily) < 60:
            return None

        current_price = safe_float(daily["Close"].iloc[-1])

        if current_price is None:
            return None

        if not (MIN_PRICE <= current_price <= MAX_PRICE):
            return None

        rally = detect_recent_rally(daily)

        if rally is None:
            return None

        # لا نريد سهمًا ما زال في منتصف الصعود.
        if current_price > rally["high_price"] * 1.03:
            return None

        support = calculate_support(daily, rally)

        if support is None or support <= 0:
            return None

        if not support_not_broken(daily, rally, support):
            return None

        support_result = analyze_support_retests(
            daily,
            rally,
            support,
        )

        # يجب أن يكون السهم قد عاد إلى منطقة الدعم
        if not price_near_support(current_price, support):
            # يسمح بالتحرك فوق الدعم قليلًا إذا كان قد اختبره مرتين
            # وسيعمل 4H لاحقًا كتأكيد.
            if not support_result["second_test"]:
                return None

        # بيانات 1H لبناء 4H
        intraday_raw = yf.download(
            ticker,
            period=INTRADAY_PERIOD,
            interval=INTRADAY_INTERVAL,
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        intraday_1h = normalize_columns(intraday_raw)
        if intraday_1h.empty:
            return None

        data_4h = build_4h(intraday_1h)
        tech = analyze_4h(data_4h)

        if not tech["ready"]:
            return None

        # إذا كان RSI مرتفعًا جدًا نرفض الإشارة.
        if tech["rsi"] is None or tech["rsi"] > RSI_MAX:
            return None

        # يجب أن يكون هناك تحسن واحد على الأقل في RSI أو MACD.
        momentum_improving = (
            tech["rsi_improving"]
            or tech["macd_improving"]
            or tech["macd_cross"]
        )

        if not momentum_improving:
            return None

        # المقاومة والأهداف
        resistances = find_resistances(
            daily,
            current_price,
            rally["high_price"],
        )

        targets = build_targets(
            current_price,
            rally["base_price"],
            rally["high_price"],
            resistances,
        )

        if not targets:
            return None

        # الدخول: منطقة حول الدعم، وليس سعرًا واحدًا.
        entry_low = max(
            support * 0.98,
            current_price * 0.97,
        )

        entry_high = min(
            support * 1.12,
            current_price * 1.03,
        )

        if entry_low >= entry_high:
            entry_low = support * 0.98
            entry_high = support * 1.08

        stop_loss = support * SUPPORT_BREAK_MULT

        # بيانات إضافية
        float_short = get_float_short(ticker)
        float_shares = float_short["float"]
        short_shares = float_short["short"]

        # Float إذا كان معروفًا يجب أن يكون داخل النطاق.
        if float_shares is not None:
            if float_shares < MIN_FLOAT_SHARES:
                return None
            if float_shares > MAX_FLOAT_SHARES:
                return None

        # تأكيد 15 دقيقة
        small = confirm_15m(ticker)

        # ----------------------------------------------------
        # نظام نقاط الدخول
        # ----------------------------------------------------
        score = 0
        reasons = []

        # الدعم
        if support_result["second_test"]:
            score += 2
            reasons.append("اختباران للدعم")

        if support_result["support_stable"]:
            score += 1
            reasons.append("الدعم مستقر")

        # RSI
        if tech["rsi_improving"]:
            score += 1
            reasons.append("RSI يتحسن")

        # MACD
        if tech["macd_positive"]:
            score += 1
            reasons.append("MACD إيجابي")

        if tech["macd_improving"] or tech["macd_cross"]:
            score += 1
            reasons.append("MACD يتحسن/تقاطع إيجابي")

        # المتوسطات
        if tech["above_ma20"]:
            score += 1
            reasons.append("فوق MA20")

        if tech["above_ma50"]:
            score += 1
            reasons.append("فوق MA50")

        # حجم
        if tech["volume_confirmed"]:
            score += 1
            reasons.append("حجم داعم")

        # 15 دقيقة
        if small["confirmed"]:
            score += 2
            reasons.append("تأكيد 15 دقيقة")

        # short
        if short_shares is not None and short_shares < MAX_SHORT_SHARES:
            score += 1
            reasons.append("Short منخفض")

        # خبر إيجابي
        catalysts = get_positive_catalysts(ticker)

        if catalysts:
            score += 1
            reasons.append("محفز إيجابي")

        # لا ندخل إلا إذا اجتمعت إشارات كافية.
        if score < MIN_ENTRY_SCORE:
            return None

        # شرط مهم: لا يكفي أن يكون المؤشر إيجابيًا؛ نريد
        # تأكيد 15 دقيقة أو MACD 4H صاعد.
        if not (
            small["confirmed"]
            or tech["macd_cross"]
            or (
                tech["macd_positive"]
                and tech["rsi_improving"]
            )
        ):
            return None

        result = {
            "ticker": ticker,
            "price": current_price,
            "rally_low": rally["base_price"],
            "rally_high": rally["high_price"],
            "rally_percent": rally["rally_percent"],
            "rally_sessions": rally["sessions"],
            "rally_start_date": rally["start_date"],
            "rally_high_date": rally["high_date"],
            "support": support,
            "support_tests": support_result["tests"],
            "successful_tests": support_result["successful_tests"],
            "entry_low": entry_low,
            "entry_high": entry_high,
            "stop_loss": stop_loss,
            "targets": targets,
            "resistances": resistances,
            "rsi": tech["rsi"],
            "rsi_improving": tech["rsi_improving"],
            "macd_positive": tech["macd_positive"],
            "macd_improving": tech["macd_improving"],
            "macd_cross": tech["macd_cross"],
            "volume_ratio": tech["volume_ratio"] or 0,
            "above_ma20": tech["above_ma20"],
            "above_ma50": tech["above_ma50"],
            "float_shares": float_shares,
            "short_shares": short_shares,
            "catalysts": catalysts,
            "small_confirmed": small["confirmed"],
            "score": score,
            "reasons": reasons,
        }

        return result

    except Exception as exc:
        print(f"[{ticker}] خطأ فردي: {exc}", flush=True)
        return None


# ============================================================
# فحص دفعة
# ============================================================

def scan_chunk(tickers, chunk_number, total_chunks):
    print(
        f"[دفعة {chunk_number}/{total_chunks}] فحص {len(tickers)} سهم...",
        flush=True,
    )

    for ticker in tickers:
        result = analyze_stock(ticker)

        if result is None:
            continue

        key = (
            ticker,
            str(result["rally_high_date"]),
            round(result["entry_low"], 3),
        )

        print(
            f"⭐ إشارة مرشحة: {ticker} | "
            f"نقاط={result['score']} | "
            f"السعر={fmt_price(result['price'])} | "
            f"الدعم={fmt_price(result['support'])}",
            flush=True,
        )

        # تنبيه واحد لنفس الفرصة.
        if key in alerted_entries:
            continue

        if send_telegram_entry(result):
            alerted_entries.add(key)


# ============================================================
# الفحص الكامل
# ============================================================

def run_scan():
    started = time.time()

    print("=" * 60, flush=True)
    print(
        f"بدء الفحص: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        flush=True,
    )

    tickers = get_all_tickers()

    if not tickers:
        print("❌ لم يتم تحميل قائمة الأسهم.", flush=True)
        return

    print(
        f"تم تحميل {len(tickers)} سهم. بدء التحليل...",
        flush=True,
    )

    total_chunks = (len(tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i:i + CHUNK_SIZE]
        scan_chunk(
            chunk,
            (i // CHUNK_SIZE) + 1,
            total_chunks,
        )

    elapsed = time.time() - started

    print(
        f"انتهى الفحص خلال {elapsed / 60:.1f} دقيقة.",
        flush=True,
    )
    print("=" * 60, flush=True)


# ============================================================
# التشغيل
# ============================================================

def main():
    # GitHub Actions ينفذ الفحص مرة واحدة في كل Job.
    # لذلك نترك loop فقط عند التشغيل اليدوي/السيرفر المستمر.
    run_scan()


if __name__ == "__main__":
    main()
