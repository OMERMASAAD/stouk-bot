# -*- coding: utf-8 -*-

"""
============================================================
PULLBACK RECOVERY RADAR
نظام مراقبة الأسهم التي صعدت بقوة ثم صححت وبدأت بالتعافي
============================================================

الفكرة:

1) السعر بين 1$ و5$.
2) السهم حقق صعوداً سابقاً لا يقل عن 100%.
3) الصعود حدث خلال 4 إلى 20 جلسة.
4) الصعود حديث، وأقصى عمر للقمة 31 يوماً.
5) نراقب رجوع السهم إلى دعم بداية الصعود.
6) ننتظر اختبار الدعم ثم إعادة اختباره.
7) نريد استقراراً وليس مجرد سقوط للسهم.
8) تحليل الاتجاه على 4 ساعات.
9) تأكيد الدخول على 15 دقيقة.
10) RSI يتحسن.
11) MACD إيجابي أو يتحسن/يتقاطع إيجابياً.
12) حجم التصحيح يكون هادئاً نسبياً.
13) المتوسطات المتحركة عامل تأكيد.
14) الأخبار الإيجابية عامل مساعد.
15) Short أقل من 50 ألف عامل إيجابي وليس شرطاً قاتلاً.
16) Fibonacci + المقاومات لتحديد الأهداف.
17) الدخول على دفعات.
18) Telegram فقط بعد اكتمال إشارة الدخول.
19) لا يتم إرسال تنبيهات أثناء مرحلة المراقبة.
20) الفحص يتم كل ساعة من GitHub Actions.

هذه أداة فلترة فنية وليست ضماناً للربح.
"""

import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
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

# الصعود السابق
MIN_RALLY_PERCENT = 100.0

# مدة الصعود
MIN_RALLY_SESSIONS = 4
MAX_RALLY_SESSIONS = 20

# أقصى عمر للصعود
MAX_RALLY_AGE_DAYS = 31

# الفلوت والـ Short عوامل مساعدة
MIN_FLOAT = 5_000_000
MAX_FLOAT = 500_000_000
MAX_SHORT_SHARES = 50_000

# منطقة الدعم
SUPPORT_ZONE_LOW = 0.90
SUPPORT_ZONE_HIGH = 1.12

# كسر الدعم
SUPPORT_BREAK_MULTIPLIER = 0.97

# عدد اختبارات الدعم
MIN_SUPPORT_TESTS = 2

# ارتداد أدنى من الاختبار
MIN_REBOUND_PERCENT = 3.0

# RSI
RSI_MAX_ENTRY = 68.0
RSI_MIN_IMPROVEMENT = 1.5

# حجم التداول
QUIET_VOLUME_RATIO = 1.60
ENTRY_VOLUME_RATIO = 1.05

# عدد الأسهم في دفعة التحميل
CHUNK_SIZE = 100

# البيانات
DAILY_PERIOD = "120d"
HOURLY_PERIOD = "30d"
SMALL_PERIOD = "10d"


# ============================================================
# الذاكرة أثناء تشغيل واحد
# ============================================================

ALERTED = set()


# ============================================================
# أدوات مساعدة
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


def clean_columns(df):

    if df is None or df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = [
            c[0] if isinstance(c, tuple) else c
            for c in df.columns
        ]

    return df


# ============================================================
# تحميل البيانات
# ============================================================

def download_data(ticker, period, interval):

    try:

        df = yf.download(

            ticker,

            period=period,

            interval=interval,

            auto_adjust=False,

            progress=False,

            threads=False,
        )

        df = clean_columns(df)

        if df.empty:
            return pd.DataFrame()

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]

        for column in required:

            if column not in df.columns:
                return pd.DataFrame()

        df = df.dropna(
            subset=required
        )

        return df

    except Exception as e:

        print(
            f"[{ticker}] خطأ تحميل {interval}: {e}",
            flush=True
        )

        return pd.DataFrame()


# ============================================================
# RSI
# ============================================================

def calculate_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = (
        avg_gain
        / avg_loss.replace(
            0,
            np.nan
        )
    )

    return 100 - (
        100 / (1 + rs)
    )


# ============================================================
# MACD
# ============================================================

def calculate_macd(close):

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    macd_line = (
        ema12 - ema26
    )

    signal = macd_line.ewm(
        span=9,
        adjust=False
    ).mean()

    histogram = (
        macd_line - signal
    )

    return (
        macd_line,
        signal,
        histogram
    )


# ============================================================
# بناء فريم 4 ساعات من 1 ساعة
# ============================================================

def build_4h(hourly):

    if hourly.empty:
        return pd.DataFrame()

    try:

        data = hourly.copy()

        data = data.sort_index()

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
                "Close",
            ]
        )

        return result

    except Exception as e:

        print(
            f"خطأ بناء فريم 4H: {e}",
            flush=True
        )

        return pd.DataFrame()


# ============================================================
# قائمة الأسهم
# ============================================================

def get_all_tickers():

    tickers = []

    sources = [

        (
            "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            "Symbol"
        ),

        (
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
            "ACT Symbol"
        ),

    ]

    for url, column in sources:

        try:

            table = pd.read_csv(
                url,
                sep="|"
            )

            table = table[
                table["Test Issue"] == "N"
            ]

            tickers.extend(
                table[column]
                .dropna()
                .astype(str)
                .tolist()
            )

        except Exception as e:

            print(
                f"خطأ تحميل قائمة الأسهم: {e}",
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

    return sorted(
        set(clean)
    )


# ============================================================
# اكتشاف صعود 100%+
# ============================================================

def detect_recent_rally(df):

    if df.empty:
        return None

    if len(df) < 35:
        return None

    data = df.copy()

    data = data.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
        ]
    )

    if len(data) < 35:
        return None

    last_position = (
        len(data) - 1
    )

    best = None

    # نبحث عن قمم حديثة
    search_start = max(
        10,
        last_position - MAX_RALLY_AGE_DAYS
    )

    for end_pos in range(
        search_start,
        last_position + 1
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

            # قاع بداية الحركة
            start_window = data.iloc[
                max(0, start_pos - 2):
                start_pos + 1
            ]

            base = safe_float(
                start_window["Low"].min()
            )

            if base is None:
                continue

            if base <= 0:
                continue

            gain = (
                (high / base) - 1
            ) * 100

            if gain < MIN_RALLY_PERCENT:
                continue

            high_date = (
                pd.Timestamp(
                    data.index[end_pos]
                ).date()
            )

            age_days = (
                pd.Timestamp.now().date()
                - high_date
            ).days

            if age_days < 0:
                continue

            if age_days > MAX_RALLY_AGE_DAYS:
                continue

            candidate = {

                "start_pos": start_pos,

                "end_pos": end_pos,

                "base": base,

                "high": high,

                "gain": gain,

                "sessions": sessions,

                "start_date":
                    pd.Timestamp(
