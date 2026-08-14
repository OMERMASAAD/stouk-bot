# -*- coding: utf-8 -*-

"""
============================================================
   PULLBACK RECOVERY RADAR
   نظام مراقبة الأسهم التي حققت صعودًا قويًا ثم صححت
============================================================

الفكرة:

1) السهم بين $1 و $5.
2) يكتشف صعودًا سابقًا >= 100%.
3) الصعود يجب أن يكون حديثًا:
   من 4 جلسات إلى 20 جلسة يومية.
4) بعد الصعود نراقب التصحيح والعودة إلى منطقة بداية الصعود.
5) نبحث عن استقرار عند الدعم.
6) ننتظر ارتدادًا ثم إعادة اختبار للدعم.
7) لا يوجد Telegram أثناء المراقبة.
8) Telegram فقط عند تحقق دخول إيجابي.
9) التحليل الأساسي على 4H.
   يتم بناء 4H من بيانات 1H لأن Yahoo Finance لا يوفر 4H
   بشكل مباشر عبر yfinance.
10) فريم 15 دقيقة يستخدم لتأكيد الدخول.
11) RSI + MACD + Volume + MA + Fibonacci + Resistance.
12) Short أقل من 50 ألف يعتبر عامل إيجابي.
13) المحفزات الإخبارية الإيجابية عامل مساعد.
14) الدخول على دفعات.
15) الأهداف من Fibonacci والمقاومات.
16) وقف الخسارة أسفل منطقة الدعم.

مهم:
هذه أداة فلترة ومراقبة وليست ضمانًا للربح.
"""

import os
import sys
import time
import warnings
import requests
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta

warnings.filterwarnings("ignore")


# ============================================================
# TELEGRAM
# ============================================================

TELEGRAM_TOKEN = (
    os.environ.get("TELEGRAM_TOKEN")
    or "ضع_التوكن_هنا"
)

TELEGRAM_CHAT_ID = (
    os.environ.get("TELEGRAM_CHAT_ID")
    or "ضع_الشات_آيدي_هنا"
)


# ============================================================
# الإعدادات الرئيسية
# ============================================================

MIN_PRICE = 1.00
MAX_PRICE = 5.00

# الصعود المطلوب
MIN_RALLY_PERCENT = 100.0

# الصعود يجب أن يكون بين 4 و20 جلسة
MIN_RALLY_SESSIONS = 4
MAX_RALLY_SESSIONS = 20

# بعد اكتشاف الصعود، نراقب الفرصة لفترة محدودة
MAX_WATCH_DAYS = 31

# حجم التداول ليس شرطًا أساسيًا للصعود
# لكنه يستخدم كعامل تأكيد
MIN_RALLY_VOLUME = 1_000_000

# Short
MAX_SHORT_SHARES = 50_000

# Float
MIN_FLOAT_SHARES = 5_000_000
MAX_FLOAT_SHARES = 500_000_000

# منطقة الدعم حول قاع بداية الصعود
SUPPORT_ZONE_LOW = 0.90
SUPPORT_ZONE_HIGH = 1.12

# هامش كسر الدعم
SUPPORT_BREAK_MULTIPLIER = 0.97

# يجب أن يكون هناك اختباران للدعم
MIN_SUPPORT_RETESTS = 2

# الارتداد الأدنى المطلوب بعد اختبار الدعم
MIN_REBOUND_PERCENT = 3.0

# حجم هادئ أثناء التصحيح
QUIET_VOLUME_RATIO = 1.50

# حجم أفضل عند الارتداد
ENTRY_VOLUME_RATIO = 1.20

# RSI
RSI_MAX_ENTRY = 65
RSI_MIN_IMPROVEMENT = 2.0

# تشغيل الفحص كل ساعة
SCAN_INTERVAL_SECONDS = 60 * 60

# حجم دفعات yfinance
CHUNK_SIZE = 100

# عدد الأسهم المعروضة داخليًا
MAX_INTERNAL_CANDIDATES = 100


# ============================================================
# متغيرات الذاكرة
# ============================================================

alerted_entries = set()

internal_candidates = {}

last_scan_time = None


# ============================================================
# أدوات عامة
# ============================================================

def safe_float(value):

    try:

        if value is None:
            return None

        value = float(value)

        if np.isnan(value):
            return None

        if np.isinf(value):
            return None

        return value

    except Exception:

        return None


def fmt_price(value):

    value = safe_float(value)

    if value is None:
        return "غير متوفر"

    return f"${value:.2f}"


def fmt_price4(value):

    value = safe_float(value)

    if value is None:
        return "غير متوفر"

    return f"${value:.4f}"


def fmt_number(value):

    value = safe_float(value)

    if value is None:
        return "غير متوفر"

    return f"{value:,.0f}"


def fmt_millions(value):

    value = safe_float(value)

    if value is None:
        return "غير متوفر"

    return f"{value / 1_000_000:.1f} مليون"


# ============================================================
# قائمة الأسهم
# ============================================================

def get_all_tickers():

    tickers = []

    try:

        nasdaq = pd.read_csv(
            "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            sep="|"
        )

        nasdaq = nasdaq[
            nasdaq["Test Issue"] == "N"
        ]

        tickers.extend(
            nasdaq["Symbol"]
            .dropna()
            .astype(str)
            .tolist()
        )

    except Exception as e:

        print(
            f"خطأ تحميل NASDAQ: {e}",
            flush=True
        )

    try:

        other = pd.read_csv(
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
            sep="|"
        )

        other = other[
            other["Test Issue"] == "N"
        ]

        tickers.extend(
            other["ACT Symbol"]
            .dropna()
            .astype(str)
            .tolist()
        )

    except Exception as e:

        print(
            f"خطأ تحميل NYSE/AMEX: {e}",
            flush=True
        )

    clean = []

    for ticker in tickers:

        ticker = ticker.strip().upper()

        if not ticker:
            continue

        if not ticker.isalpha():
            continue

        if len(ticker) > 5:
            continue

        clean.append(ticker)

    return sorted(set(clean))


# ============================================================
# المؤشرات
# ============================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi


def calculate_macd(close):

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    histogram = macd - signal

    return macd, signal, histogram


# ============================================================
# تحويل 1H إلى 4H
# ============================================================

def build_4h(df):

    if df is None or df.empty:
        return pd.DataFrame()

    data = df.copy()

    data = data.sort_index()

    required = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in required:

        if col not in data.columns:
            return pd.DataFrame()

    try:

        result = data.resample(
            "4h"
        ).agg({

            "Open": "first",

            "High": "max",

            "Low": "min",

            "Close": "last",

            "Volume": "sum",
        })

        result = result.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close"
            ]
        )

        return result

    except Exception:

        return pd.DataFrame()


# ============================================================
# تحليل Fibonacci
# ============================================================

def fibonacci_levels(low, high):

    low = safe_float(low)
    high = safe_float(high)

    if low is None or high is None:
        return {}

    if high <= low:
        return {}

    diff = high - low

    return {

        "0.236": high - diff * 0.236,

        "0.382": high - diff * 0.382,

        "0.500": high - diff * 0.500,

        "0.618": high - diff * 0.618,

        "0.786": high - diff * 0.786,

    }


def fibonacci_targets(low, high):

    low = safe_float(low)
    high = safe_float(high)

    if low is None or high is None:
        return []

    diff = high - low

    if diff <= 0:
        return []

    return [

        high + diff * 0.272,

        high + diff * 0.618,

        high + diff * 1.000,

        high + diff * 1.618,

    ]


# ============================================================
# اكتشاف الصعود القوي
# ============================================================

def detect_recent_rally(df):

    if df is None or df.empty:
        return None

    if len(df) < MAX_RALLY_SESSIONS + 5:
        return None

    data = df.copy()

    data = data.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]
    )

    if data.empty:
        return None

    # نبحث عن صعود حديث
    # من 4 إلى 20 جلسة
    recent_end = len(data) - 1

    # لا نريد استخدام آخر يوم فقط بطريقة ضيقة
    search_start = max(
        0,
        len(data) - MAX_WATCH_DAYS - 5
    )

    best = None

    for end_pos in range(
        search_start,
        recent_end + 1
    ):

        high = safe_float(
            data.iloc[end_pos]["High"]
        )

        if high is None:
            continue

        for sessions in range(
            MIN_RALLY_SESSIONS,
            MAX_RALLY_SESSIONS + 1
        ):

            start_pos = (
                end_pos - sessions
            )

            if start_pos < 0:
                continue

            start_low = safe_float(
                data.iloc[start_pos]["Low"]
            )

            start_close = safe_float(
                data.iloc[start_pos]["Close"]
            )

            if (
                start_low is None
                or start_low <= 0
            ):
                continue

            # نستخدم أقل سعر في بداية الحركة
            base = min(
                start_low,
                start_close
                if start_close is not None
                else start_low
            )

            rally_percent = (
                (high - base)
                / base
            ) * 100

            if rally_percent < MIN_RALLY_PERCENT:
                continue

            start_date = (
                pd.Timestamp(
                    data.index[start_pos]
                ).date()
            )

            high_date = (
                pd.Timestamp(
                    data.index[end_pos]
                ).date()
            )

            today = datetime.now().date()

            age_days = (
                today - high_date
            ).days

            if age_days > MAX_WATCH_DAYS:
                continue

            # متوسط حجم الصعود
            rally_volume = safe_float(
                data.iloc[
                    start_pos:end_pos + 1
                ]["Volume"].mean()
            )

            candidate = {

                "start_pos": start_pos,

                "end_pos": end_pos,

                "start_date": start_date,

                "high_date": high_date,

                "base_price": base,

                "high_price": high,

                "rally_percent": rally_percent,

                "sessions": sessions,

                "rally_volume": rally_volume,

            }

            # نفضل أحدث صعود قوي
            if best is None:

                best = candidate

            else:

                if candidate["high_date"] > best["high_date"]:

                    best = candidate

                elif (
                    candidate["high_date"]
                    == best["high_date"]
                    and candidate["rally_percent"]
                    > best["rally_percent"]
                ):

                    best = candidate

    return best


# ============================================================
# تحديد الدعم قبل الصعود
# ============================================================

def calculate_rally_support(
    df,
    rally
):

    if rally is None:
        return None

    start_pos = rally["start_pos"]

    # نأخذ المنطقة السابقة لبداية الصعود
    begin = max(
        0,
        start_pos - 5
    )

    end = min(
        len(df),
        start_pos + 2
    )

    window = df.iloc[
        begin:end
    ]

    if window.empty:
        return None

    lows = window["Low"].dropna()

    if lows.empty:
        return None

    support = float(
        lows.min()
    )

    return support


# ============================================================
# دعم قريب؟
# ============================================================

def is_near_support(
    price,
    support
):

    price = safe_float(price)
    support = safe_float(support)

    if (
        price is None
        or support is None
        or support <= 0
    ):

        return False

    lower = (
        support
        * SUPPORT_ZONE_LOW
    )

    upper = (
        support
        * SUPPORT_ZONE_HIGH
    )

    return (
        lower
        <= price
        <= upper
    )


# ============================================================
# تحليل اختبارات الدعم
# ============================================================

def analyze_support_retests(
    df,
    support,
    rally
):

    result = {

        "tests": 0,

        "successful_tests": 0,

        "last_test": None,

        "last_rebound": None,

        "support_stable": False,

        "second_test": False,

    }

    if (
        df is None
        or df.empty
        or support is None
        or rally is None
    ):

        return result

    start_date = rally["start_date"]

    after = df[
        df.index.date >= start_date
    ].copy()

    if after.empty:
        return result

    lower = (
        support
        * SUPPORT_ZONE_LOW
    )

    upper = (
        support
        * SUPPORT_ZONE_HIGH
    )

    tests = []

    for idx, row in after.iterrows():

        low = safe_float(
            row["Low"]
        )

        close = safe_float(
            row["Close"]
        )

        if low is None:
            continue

        if not (
            lower
            <= low
            <= upper
        ):

            continue

        rebound = False

        if close is not None:

            rebound = (
                close
                >= low
                * (
                    1
                    + MIN_REBOUND_PERCENT / 100
                )
            )

        current_date = (
            pd.Timestamp(idx).date()
        )

        # فصل الاختبارات
        if not tests:

            tests.append({

                "date": current_date,

                "low": low,

                "rebound": rebound

            })

        else:

            previous = tests[-1]

            gap = (
                current_date
                - previous["date"]
            ).days

            if gap >= 2:

                tests.append({

                    "date": current_date,

                    "low": low,

                    "rebound": rebound

                })

            elif rebound:

                previous["rebound"] = True

    result["tests"] = len(tests)

    result["successful_tests"] = sum(
        x["rebound"]
        for x in tests
    )

    if tests:

        result["last_test"] = tests[-1]["date"]

        result["last_rebound"] = (
            tests[-1]["rebound"]
        )

    if (
        result["tests"]
        >= MIN_SUPPORT_RETESTS
    ):

        result["second_test"] = True

    if (
        result["successful_tests"]
        >= 1
    ):

        result["support_stable"] = True

    return result


# ============================================================
# RSI / MACD / Volume على 4H
# ============================================================

def technical_analysis_4h(df):

    result = {

        "rsi": None,

        "previous_rsi": None,

        "rsi_improving": False,

        "macd": None,

        "signal": None,

        "hist": None,

        "previous_hist": None,

        "macd_positive": False,

        "macd_improving": False,

        "macd_cross": False,

        "volume": None,

        "volume_avg": None,

        "volume_ratio": None,

        "quiet_volume": False,

        "ma20": None,

        "ma50": None,

        "above_ma20": False,

        "above_ma50": False,

    }

    if df is None or len(df) < 55:
        return result

    data = df.copy()

    close = data["Close"]

    data["RSI"] = calculate_rsi(
        close
    )

    (
        data["MACD"],
        data["MACD_SIGNAL"],
        data["MACD_HIST"]
    ) = calculate_macd(
        close
    )

    data["MA20"] = (
        close.rolling(20).mean()
    )

    data["MA50"] = (
        close.rolling(50).mean()
    )

    latest = data.iloc[-1]

    rsi = safe_float(
        latest["RSI"]
    )

    previous_rsi = safe_float(
        data["RSI"].iloc[-2]
    )

    macd = safe_float(
        latest["MACD"]
    )

    signal = safe_float(
        latest["MACD_SIGNAL"]
    )

    hist = safe_float(
        latest["MACD_HIST"]
    )

    previous_hist = safe_float(
        data["MACD_HIST"].iloc[-2]
    )

    volume = safe_float(
        latest["Volume"]
    )

    volume_avg = safe_float(
        data["Volume"].tail(20).mean()
    )

    ma20 = safe_float(
        latest["MA20"]
    )

    ma50 = safe_float(
        latest["MA50"]
    )

    result.update({

        "rsi": rsi,

        "previous_rsi": previous_rsi,

        "rsi_improving": (
            rsi is not None
            and previous_rsi is not None
            and rsi > previous_rsi
            and (
                rsi - previous_rsi
                >= RSI_MIN_IMPROVEMENT
            )
        ),

        "macd": macd,

        "signal": signal,

        "hist": hist,

        "previous_hist": previous_hist,

        "macd_positive": (
            macd is not None
            and signal is not None
            and macd > signal
        ),

        "macd_improving": (
            hist is not None
            and previous_hist is not None
            and hist > previous_hist
        ),

        "macd_cross": False,

        "volume": volume,

        "volume_avg": volume_avg,

        "ma20": ma20,

        "ma50": ma50,

        "above_ma20": (
            ma20 is not None
            and close.iloc[-1] >= ma20
        ),

        "above_ma50": (
            ma50 is not None
            and close.iloc[-1] >= ma50
        ),
    })

    # تقاطع MACD
    if len(data) >= 3:

        m1 = safe_float(
            data["MACD"].iloc[-1]
        )

        s1 = safe_float(
            data["MACD_SIGNAL"].iloc[-1]
        )

        m2 = safe_float(
            data["MACD"].iloc[-2]
        )

        s2 = safe_float(
            data["MACD_SIGNAL"].iloc[-2]
        )

        result["macd_cross"] = (

            m1 is not None
            and s1 is not None
            and m2 is not None
            and s2 is not None

            and m1 > s1

            and m2 <= s2
        )

    if (
        volume is not None
        and volume_avg is not None
        and volume_avg > 0
    ):

        ratio = (
            volume
            / volume_avg
        )

        result["volume_ratio"] = ratio

        result["quiet_volume"] = (
            ratio <= QUIET_VOLUME_RATIO
        )

    return result


# ============================================================
# تحليل الفريم الصغير 15 دقيقة
# ============================================================

def small_timeframe_confirmation(
    ticker
):

    result = {

        "confirmed": False,

        "price": None,

        "rsi": None,

        "rsi_improving": False,

        "macd_positive": False,

        "volume_confirmed": False,

        "reason": "",
    }

    try:

        df = yf.download(
            ticker,
            period="10d",
            interval="15m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return result

        if isinstance(
            df.columns,
            pd.MultiIndex
        ):

            df.columns = [
                x[0]
                for x in df.columns
            ]

        df = df.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        )

        if len(df) < 40:
            return result

        close = df["Close"]

        df["RSI"] = calculate_rsi(
            close
        )

        (
            df["MACD"],
            df["MACD_SIGNAL"],
            df["MACD_HIST"]
        ) = calculate_macd(
            close
        )

        latest = df.iloc[-1]

        price = safe_float(
            latest["Close"]
        )

        rsi = safe_float(
            latest["RSI"]
        )

        previous_rsi = safe_float(
            df["RSI"].iloc[-2]
        )

        macd = safe_float(
            latest["MACD"]
        )

        macd_signal = safe_float(
            latest["MACD_SIGNAL"]
        )

        volume = safe_float(
            latest["Volume"]
        )

        avg_volume = safe_float(
            df["Volume"].tail(20).mean()
        )

        result["price"] = price

        result["rsi"] = rsi

        result["rsi_improving"] = (
            rsi is not None
            and previous_rsi is not None
            and rsi > previous_rsi
        )

        result["macd_positive"] = (
            macd is not None
            and macd_signal is not None
            and macd > macd_signal
        )

        result["volume_confirmed"] = (
            volume is not None
            and avg_volume is not None
            and avg_volume > 0
            and volume >= avg_volume * 1.05
        )

        # لا نطلب جميع المؤشرات بشكل صارم
        # حتى لا تصبح الاستراتيجية مستحيلة
        confirmations = 0

        if result["rsi_improving"]:
            confirmations += 1

        if result["macd_positive"]:
            confirmations += 1

        if result["volume_confirmed"]:
            confirmations += 1

        if (
            price is not None
            and confirmations >= 2
        ):

            result["confirmed"] = True

            result["reason"] = (
                "تأكيد إيجابي على الفريم الصغير"
            )

        else:

            result["reason"] = (
                "لم يكتمل تأكيد الفريم الصغير"
            )

        return result

    except Exception as e:

        result["reason"] = str(e)

        return result


# ============================================================
# المقاومة
# ============================================================

def find_resistances(
    df,
    current_price,
    rally_high
):

    levels = []

    if df is None or df.empty:
        return levels

    current_price = safe_float(
        current_price
    )

    rally_high = safe_float(
        rally_high
    )

    if current_price is None:
        return levels

    # قمم سابقة
    highs = (
        df["High"]
        .dropna()
        .tail(60)
        .sort_values(
            ascending=False
        )
    )

    for value in highs:

        value = safe_float(value)

        if value is None:
            continue

        if value <= current_price * 1.02:
            continue

        if value > current_price * 5:
            continue

        # منع التكرار
        if not any(
            abs(value - x)
            / x
            < 0.03
            for x in levels
        ):

            levels.append(value)

    # قمة الصعود
    if (
        rally_high is not None
        and rally_high
        > current_price * 1.02
    ):

        if not any(
            abs(rally_high - x)
            / x
            < 0.03
            for x in levels
        ):

            levels.append(rally_high)

    return sorted(levels)


# ============================================================
# الأخبار / المحفزات
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

    # عربي
    "موافقة",
    "عقد",
    "شراكة",
    "استحواذ",
    "اندماج",
    "أرباح",
    "نتائج",
]


def get_positive_catalysts(
    ticker
):

    catalysts = []

    try:

        stock = yf.Ticker(
            ticker
        )

        news = stock.news

        if not news:
            return catalysts

        for item in news[:10]:

            title = ""

            try:

                content = item.get(
                    "content",
                    {}
                )

                if isinstance(
                    content,
                    dict
                ):

                    title = (
                        content.get(
                            "title"
                        )
                        or ""
                    )

            except Exception:
                pass

            if not title:

                title = str(
                    item.get(
                        "title",
                        ""
                    )
                )

            title_lower = title.lower()

            if any(
                keyword.lower()
                in title_lower
                for keyword
                in POSITIVE_KEYWORDS
            ):

                catalysts.append(
                    title
                )

    except Exception:
        pass

    return catalysts[:3]


# ============================================================
# بيانات Float / Short
# ============================================================

def get_float_short(
    ticker
):

    result = {

        "float": None,

        "short": None,

    }

    try:

        stock = yf.Ticker(
            ticker
        )

        info = stock.info

        result["float"] = safe_float(
            info.get(
                "floatShares"
            )
        )

        result["short"] = safe_float(
            info.get(
                "sharesShort"
            )
        )

    except Exception:
        pass

    return result


# ============================================================
# بناء الأهداف
# ============================================================

def build_targets(
    current_price,
    rally_low,
    rally_high,
    resistances
):

    targets = []

    fib_targets = fibonacci_targets(
        rally_low,
        rally_high
    )

    for target in fib_targets:

        target = safe_float(
            target
        )

        if (
            target is not None
            and target > current_price * 1.05
        ):

            targets.append(
                target
            )

    for resistance in resistances:

        resistance = safe_float(
            resistance
        )

        if (
            resistance is not None
            and resistance > current_price * 1.05
        ):

            targets.append(
                resistance
            )

    unique = []

    for target in sorted(
        targets
    ):

        if not any(
            abs(target - x)
            / x
            < 0.04
            for x in unique
        ):

            unique.append(target)

    return unique[:5]


# ============================================================
# إرسال Telegram
# ============================================================

def send_telegram_entry(
    result
):

    if (
        TELEGRAM_TOKEN
        == "ضع_التوكن_هنا"
        or TELEGRAM_CHAT_ID
        == "ضع_الشات_آيدي_هنا"
    ):

        print(
            "تحذير: بيانات Telegram غير موجودة."
        )

        return False

    ticker = result["ticker"]

    targets = result["targets"]

    target_lines = []

    for i, target in enumerate(
        targets,
        1
    ):

        target_lines.append(
            f"🎯 الهدف {i}: {fmt_price(target)}"
        )

    targets_text = "\n".join(
        target_lines
    )

    catalysts = result[
        "catalysts"
    ]

    if catalysts:

        catalyst_text = (
            "📰 محفزات إيجابية:\n"
            + "\n".join(
                f"• {x}"
                for x in catalysts
            )
        )

    else:

        catalyst_text = (
            "📰 المحفزات: لا يوجد خبر إيجابي واضح حاليًا"
        )

    message = (

        "🚨🚨 إشارة دخول مؤكدة 🚨🚨\n"
        "\n"

        f"📌 السهم: {ticker}\n"

        f"💵 السعر الحالي: {fmt_price(result['price'])}\n"

        "\n"

        "📈 سبب المتابعة:\n"
        f"• صعود سابق: +{result['rally_percent']:.1f}%\n"
        f"• مدة الصعود: {result['rally_sessions']} جلسات\n"
        f"• قاع الصعود: {fmt_price(result['rally_low'])}\n"
        f"• قمة الصعود: {fmt_price(result['rally_high'])}\n"

        "\n"

        "🟢 منطقة الدخول على دفعات:\n"
        f"• من {fmt_price(result['entry_low'])}"
        f" إلى {fmt_price(result['entry_high'])}\n"

        "\n"

        f"🛡️ الدعم: {fmt_price(result['support'])}\n"

        f"⛔ وقف الخسارة: {fmt_price(result['stop_loss'])}\n"

        "\n"

        f"{targets_text}\n"

        "\n"

        "📊 تأكيدات الدخول:\n"

        f"• RSI: {result['rsi']:.1f}\n"
        f"• RSI يتحسن: {'نعم' if result['rsi_improving'] else 'لا'}\n"
        f"• MACD إيجابي: {'نعم' if result['macd_positive'] else 'لا'}\n"
        f"• MACD يتحسن: {'نعم' if result['macd_improving'] else 'لا'}\n"
        f"• حجم التداول: {result['volume_ratio']:.2f}x من المتوسط\n"
        f"• اختبارات الدعم: {result['support_tests']}\n"
        f"• اختبارات ناجحة: {result['successful_tests']}\n"

        "\n"

        f"💎 الفلوت: {fmt_millions(result['float_shares'])}\n"
        f"📉 البيع على المكشوف: {fmt_number(result['short_shares'])}\n"

        "\n"

        f"{catalyst_text}\n"

        "\n"

        "📌 طريقة الدخول:\n"
        "دخول تدريجي على دفعات، وليس بكامل السيولة من أول سعر.\n"

        "\n"

        "⏱️ المتابعة:\n"
        "التحليل على 4 ساعات، والتأكيد على الفريم الأصغر.\n"

        "\n"

        "⚠️ تنبيه آلي للمراقبة الفنية وليس توصية مالية. "
        "راجع الخبر والشارت وإدارة المخاطر قبل الدخول."
    )

    url = (
        "https://api.telegram.org/"
        f"bot{TELEGRAM_TOKEN}/sendMessage"
    )

    try:

        response = requests.post(

            url,

            data={

                "chat_id":
                    TELEGRAM_CHAT_ID,

                "text":
                    message,

            },

            timeout=20
        )

        if response.ok:

            print(
                f"✅ تم إرسال تنبيه دخول: {ticker}",
                flush=True
            )

            return True

        print(
            f"❌ فشل Telegram: {response.text}",
            flush=True
        )

        return False

    except Exception as e:

        print(
            f"❌ خطأ Telegram: {e}",
            flush=True
        )

        return False


# ============================================================
# تحليل السهم
# ============================================================

def analyze_stock(
    ticker
):

    try:

        # ----------------------------------------------------
        # البيانات اليومية
        # ----------------------------------------------------

        daily = yf.download(

            ticker,

            period="120d",

            interval="1d",

            auto_adjust=False,

            progress=False,

            threads=False,
        )

        if daily is None or daily.empty:
            return None

        if isinstance(
            daily.columns,
            pd.MultiIndex
        ):

            daily.columns = [
                x[0]
                for x in daily.columns
            ]

        daily = daily.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
                "Volume"
            ]
        )

        if len(daily) < 30:
            return None

        current
