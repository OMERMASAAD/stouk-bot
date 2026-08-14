# -*- coding: utf-8 -*-
"""
PULLBACK RECOVERY RADAR
يراقب الأسهم التي صعدت 100%+ ثم صححت إلى دعم بداية الصعود.
لا يرسل Telegram أثناء المراقبة؛ يرسل فقط عند اكتمال إشارة دخول إيجابية.
التحليل الرئيسي 4H مبني من 1H، والتأكيد النهائي 15m.
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

# =========================
# Telegram
# =========================
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

# =========================
# الاستراتيجية
# =========================
MIN_PRICE = 1.0
MAX_PRICE = 5.0

MIN_RALLY_PERCENT = 100.0
MIN_RALLY_SESSIONS = 4
MAX_RALLY_SESSIONS = 20
MAX_RALLY_AGE_DAYS = 31

DAILY_PERIOD = "180d"
ONE_HOUR_PERIOD = "30d"
ENTRY_PERIOD = "10d"

MIN_FLOAT = 5_000_000
MAX_FLOAT = 500_000_000

# Short أقل من 50 ألف عامل إيجابي فقط
MAX_SHORT = 50_000

# منطقة الدعم
SUPPORT_LOW = 0.90
SUPPORT_HIGH = 1.12
SUPPORT_BREAK = 0.97

# نحتاج اختبارين للدعم
MIN_SUPPORT_TESTS = 2

# ارتداد من الدعم
MIN_REBOUND = 0.03

# RSI
RSI_IMPROVEMENT = 1.5
RSI_MAX = 70.0

# حجم التداول
ENTRY_VOLUME_RATIO = 1.05

# الحد الأدنى لنقاط الدخول
MIN_SCORE = 6

# حجم دفعة الفحص
CHUNK_SIZE = 100

# الأخبار الإيجابية
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

# منع تكرار نفس التنبيه
alerted = set()


# =========================
# أدوات عامة
# =========================

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

    if x is None:
        return "غير متوفر"

    return f"${x:.2f}"


def millions(x):
    x = num(x)

    if x is None:
        return "غير متوفر"

    return f"{x / 1_000_000:.1f} مليون"


def number(x):
    x = num(x)

    if x is None:
        return "غير متوفر"

    return f"{x:,.0f}"


def clean_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    if isinstance(d.columns, pd.MultiIndex):

        names = list(
            d.columns.get_level_values(0)
        )

        if all(
            c in names
            for c in [
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
            ]
        ):
            d.columns = d.columns.get_level_values(0)

        else:
            d.columns = d.columns.get_level_values(-1)

    d.columns = [
        str(c).strip()
        for c in d.columns
    ]

    needed = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(
        c in d.columns
        for c in needed
    ):
        return pd.DataFrame()

    return d.dropna(
        subset=needed
    )


# =========================
# قائمة الأسهم
# =========================

def get_all_tickers():

    out = []

    try:

        x = pd.read_csv(
            "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            sep="|",
        )

        x = x[
            x["Test Issue"] == "N"
        ]

        out += (
            x["Symbol"]
            .dropna()
            .astype(str)
            .tolist()
        )

    except Exception as e:

        print(
            "خطأ NASDAQ:",
            e,
            flush=True,
        )

    try:

        x = pd.read_csv(
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
            sep="|",
        )

        x = x[
            x["Test Issue"] == "N"
        ]

        out += (
            x["ACT Symbol"]
            .dropna()
            .astype(str)
            .tolist()
        )

    except Exception as e:

        print(
            "خطأ NYSE/AMEX:",
            e,
            flush=True,
        )

    return sorted(
        {
            x.strip().upper()
            for x in out
            if x.strip().isalpha()
            and len(x.strip()) <= 5
        }
    )


# =========================
# RSI
# =========================

def rsi(
    close,
    period=14,
):

    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    ag = gain.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    al = loss.ewm(
        alpha=1 / period,
        adjust=False,
    ).mean()

    rs = ag / al.replace(
        0,
        np.nan,
    )

    return 100 - (
        100 / (1 + rs)
    )


# =========================
# MACD
# =========================

def macd(close):

    e12 = close.ewm(
        span=12,
        adjust=False,
    ).mean()

    e26 = close.ewm(
        span=26,
        adjust=False,
    ).mean()

    m = e12 - e26

    s = m.ewm(
        span=9,
        adjust=False,
    ).mean()

    return (
        m,
        s,
        m - s,
    )


# =========================
# اكتشاف الصعود 100%+
# =========================

def detect_rally(df):

    d = clean_df(df)

    if len(d) < 60:
        return None

    today = datetime.now().date()

    best = None

    start_search = max(
        1,
        len(d) - 55,
    )

    for end in range(
        start_search,
        len(d),
    ):

        high = num(
            d.iloc[end]["High"]
        )

        if high is None or high <= 0:
            continue

        high_date = (
            pd.Timestamp(
                d.index[end]
            ).date()
        )

        age = (
            today - high_date
        ).days

        if age < 1 or age > MAX_RALLY_AGE_DAYS:
            continue

        for sessions in range(
            MIN_RALLY_SESSIONS,
            MAX_RALLY_SESSIONS + 1,
        ):

            start = end - sessions

            if start < 0:
                continue

            low = num(
                d.iloc[start]["Low"]
            )

            close_end = num(
                d.iloc[end]["Close"]
            )

            if (
                low is None
                or low <= 0
                or close_end is None
            ):
                continue

            pct = (
                (high - low)
                / low
                * 100
            )

            if pct < MIN_RALLY_PERCENT:
                continue

            # منع اعتبار spike منفصل جدًا فرصة مثالية
            if close_end < (
                low
                + (high - low) * 0.30
            ):
                continue

            candidate = {
                "start": start,
                "end": end,
                "low": low,
                "high": high,
                "percent": pct,
                "sessions": sessions,
                "start_date": pd.Timestamp(
                    d.index[start]
                ).date(),
                "high_date": high_date,
            }

            if (
                best is None
                or (
                    candidate["high_date"],
                    candidate["percent"],
                )
                > (
                    best["high_date"],
                    best["percent"],
                )
            ):
                best = candidate

    return best


# =========================
# تحديد الدعم
# =========================

def support_level(
    df,
    rally,
):

    d = clean_df(df)

    if d.empty or rally is None:
        return None

    a = max(
        0,
        rally["start"] - 5,
    )

    b = min(
        len(d),
        rally["start"] + 2,
    )

    lows = (
        d.iloc[a:b]["Low"]
        .dropna()
    )

    if lows.empty:
        return None

    return num(
        lows.min()
    )


# =========================
# اختبار الدعم
# =========================

def support_tests(
    df,
    rally,
    support,
):

    result = {
        "tests": 0,
        "successful": 0,
        "second": False,
        "stable": False,
    }

    d = clean_df(df)

    if (
        d.empty
        or rally is None
        or support is None
    ):
        return result

    after = d.iloc[
        rally["start"]:
    ]

    lo = support * SUPPORT_LOW
    hi = support * SUPPORT_HIGH

    tests = []

    for idx, row in after.iterrows():

        low = num(
            row["Low"]
        )

        close = num(
            row["Close"]
        )

        if (
            low is None
            or close is None
            or not (
                lo <= low <= hi
            )
        ):
            continue

        dt = pd.Timestamp(
            idx
        ).date()

        rebound = (
            close
            >= low
            * (1 + MIN_REBOUND)
        )

        if (
            not tests
            or (
                dt
                - tests[-1]["date"]
            ).days >= 2
        ):

            tests.append(
                {
                    "date": dt,
                    "rebound": rebound,
                }
            )

        elif rebound:

            tests[-1][
                "rebound"
            ] = True

    result["tests"] = len(
        tests
    )

    result["successful"] = sum(
        1
        for t in tests
        if t["rebound"]
    )

    result["second"] = (
        result["tests"]
        >= MIN_SUPPORT_TESTS
    )

    result["stable"] = (
        result["successful"]
        >= 1
    )

    return result


# =========================
# هل الدعم انكسر؟
# =========================

def support_not_broken(
    df,
    rally,
    support,
):

    d = clean_df(df)

    if (
        d.empty
        or rally is None
        or support is None
    ):
        return False

    closes = (
        d.iloc[
            rally["start"]:
        ]["Close"]
        .dropna()
    )

    return not (
        closes
        < support * SUPPORT_BREAK
    ).any()


# =========================
# بناء 4H من 1H
# =========================

def build_4h(df):

    d = clean_df(df)

    if d.empty:
        return pd.DataFrame()

    try:

        result = d.resample(
            "4h"
        ).agg(
            {
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum",
            }
        )

        return result.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close",
            ]
        )

    except Exception:

        return pd.DataFrame()


# =========================
# تحليل 4H
# =========================

def analyze_4h(df):

    out = {
        "ready": False,
        "rsi": None,
        "rsi_up": False,
        "macd_positive": False,
        "macd_up": False,
        "macd_cross": False,
        "volume_ratio": 0,
        "volume_ok": False,
        "ma20": False,
        "ma50": False,
    }

    d = clean_df(df)

    if len(d) < 55:
        return out

    c = d["Close"]

    d["RSI"] = rsi(c)

    (
        d["MACD"],
        d["SIGNAL"],
        d["HIST"],
    ) = macd(c)

    d["MA20"] = (
        c.rolling(20).mean()
    )

    d["MA50"] = (
        c.rolling(50).mean()
    )

    a = d.iloc[-1]
    b = d.iloc[-2]

    rv = num(a["RSI"])
    rp = num(b["RSI"])

    m = num(a["MACD"])
    s = num(a["SIGNAL"])

    h = num(a["HIST"])
    hp = num(b["HIST"])

    mp = num(b["MACD"])
    sp = num(b["SIGNAL"])

    vol = num(a["Volume"])

    av = num(
        d["Volume"]
        .tail(20)
        .mean()
    )

    out.update(
        {
            "ready": True,

            "rsi": rv,

            "rsi_up": (
                rv is not None
                and rp is not None
                and rv - rp
                >= RSI_IMPROVEMENT
            ),

            "macd_positive": (
                m is not None
                and s is not None
                and m > s
            ),

            "macd_up": (
                h is not None
                and hp is not None
                and h > hp
            ),

            "macd_cross": (
                m is not None
                and s is not None
                and mp is not None
                and sp is not None
                and m > s
                and mp <= sp
            ),

            "volume_ratio": (
                vol / av
                if (
                    vol is not None
                    and av
                    and av > 0
                )
                else 0
            ),

            "volume_ok": (
                vol is not None
                and av is not None
                and av > 0
                and vol
                >= av
                * ENTRY_VOLUME_RATIO
            ),

            "ma20": (
                num(a["MA20"])
                is not None
                and c.iloc[-1]
                >= a["MA20"]
            ),

            "ma50": (
                num(a["MA50"])
                is not None
                and c.iloc[-1]
                >= a["MA50"]
            ),
        }
    )

    return out


# =========================
# تأكيد 15 دقيقة
# =========================

def confirm_15m(
    ticker,
):

    out = {
        "confirmed": False,
        "rsi": None,
        "score": 0,
    }

    try:

        raw = yf.download(
            ticker,
            period=ENTRY_PERIOD,
            interval="15m",
            auto_adjust=False,
            progress=False,
            threads=False,
        )

        d = clean_df(raw)

        if len(d) < 50:
            return out

        c = d["Close"]

        d["RSI"] = rsi(c)

        (
            d["MACD"],
            d["SIGNAL"],
            _,
        ) = macd(c)

        a = d.iloc[-1]
        b = d.iloc[-2]

        rv = num(a["RSI"])
        rp = num(b["RSI"])

        m = num(a["MACD"])
        s = num(a["SIGNAL"])

        vol = num(
            a["Volume"]
        )

        av = num(
            d["Volume"]
            .tail(20)
            .mean()
        )

        out["rsi"] = rv

        if (
            rv is not None
            and rp is not None
            and rv > rp
        ):
            out["score"] += 1

        if (
            m is not None
            and s is not None
            and m > s
        ):
            out["score"] += 1

        if (
            vol is not None
            and av
            and av > 0
            and vol
            >= av * 1.05
        ):
            out["score"] += 1

        out["confirmed"] = (
            out["score"] >= 2
        )

    except Exception as e:

        print(
            f"[{ticker}] خطأ 15m: {e}",
            flush=True,
        )

    return out


# =========================
# Float / Short
# =========================

def get_float_short(
    ticker,
):

    try:

        info = (
            yf.Ticker(
                ticker
            ).info
        )

        return (
            num(
                info.get(
                    "floatShares"
                )
            ),
            num(
                info.get(
                    "sharesShort"
                )
            ),
        )

    except Exception:

        return (
            None,
            None,
        )


# =========================
# الأخبار الإيجابية
# =========================

def positive_news(
    ticker,
):

    out = []

    try:

        news = (
            yf.Ticker(
                ticker
            ).news
            or []
        )

        for item in news[:10]:

            title = ""

            content = item.get(
                "content",
                {},
            )

            if isinstance(
                content,
                dict,
            ):
                title = (
                    content.get(
                        "title"
                    )
                    or ""
                )

            if not title:

                title = (
                    item.get(
                        "title"
                    )
                    or ""
                )

            title = str(
                title
            )

            if any(
                k in title.lower()
                for k in POSITIVE_KEYWORDS
            ):

                out.append(
                    title
                )

    except Exception:

        pass

    return out[:3]


# =========================
# المقاومات
# =========================

def resistances(
    df,
    current,
    rally_high,
):

    d = clean_df(df)

    if d.empty:
        return []

    current = num(
        current
    )

    levels = []

    for x in (
        d["High"]
        .dropna()
        .tail(90)
    ):

        x = num(x)

        if x is None:
            continue

        if (
            x
            <= current * 1.03
        ):
            continue

        if (
            x
            > current * 4
        ):
            continue

        if not any(
            abs(x - y)
            / y
            < 0.04
            for y in levels
        ):

            levels.append(x)

    if (
        rally_high
        and rally_high
        > current * 1.03
        and not any(
            abs(
                rally_high - y
            )
            / y
            < 0.04
            for y in levels
        )
    ):

        levels.append(
            rally_high
        )

    return sorted(
        levels
    )


# =========================
# Fibonacci
# =========================

def fib_targets(
    low,
    high,
):

    diff = high - low

    if diff <= 0:
        return []

    return [
        high + diff * 0.272,
        high + diff * 0.618,
        high + diff,
        high + diff * 1.618,
    ]


def make_targets(
    current,
    low,
    high,
    levels,
):

    vals = [
        x
        for x in fib_targets(
            low,
            high,
        )
        if x
        > current * 1.05
    ]

    vals += [
        x
        for x in levels
        if x
        > current * 1.05
    ]

    vals = sorted(
        vals
    )

    out = []

    for x in vals:

        if not any(
            abs(x - y)
            / y
            < 0.04
            for y in out
        ):

            out.append(x)

    return out[:5]


# =========================
# Telegram
# =========================

def send_alert(r):

    if (
        not TELEGRAM_TOKEN
        or not TELEGRAM_CHAT_ID
    ):

        print(
            "⚠️ Telegram غير مضبوط.",
            flush=True,
        )

        return False

    targets = "\n".join(
        f"🎯 الهدف {i}: {price(x)}"
        for i, x in enumerate(
            r["targets"],
            1,
        )
    )

    news = (
        "\n".join(
            f"• {x}"
            for x in r["news"]
        )
        if r["news"]
        else
        "لا يوجد محفز إيجابي واضح في البيانات المتاحة."
    )

    short_ok = (
        r["short"] is not None
        and r["short"] < MAX_SHORT
    )

    msg = (

        "🚨 إشارة دخول إيجابية 🚨\n\n"

        f"📌 السهم: {r['ticker']}\n"

        f"💵 السعر الحالي: "
        f"{price(r['price'])}\n\n"

        "📈 الصعود السابق:\n"

        f"• الصعود: "
        f"+{r['rally_pct']:.1f}%\n"

        f"• المدة: "
        f"{r['sessions']} جلسات\n"

        f"• بداية الحركة: "
        f"{price(r['rally_low'])}\n"

        f"• قمة الحركة: "
        f"{price(r['rally_high'])}\n\n"

        "🟢 منطقة الدخول على دفعات:\n"

        f"• من "
        f"{price(r['entry_low'])}"
        f" إلى "
        f"{price(r['entry_high'])}\n"

        f"• الدعم: "
        f"{price(r['support'])}\n"

        f"• وقف الخسارة: "
        f"{price(r['stop'])}\n\n"

        f"{targets}\n\n"

        "📊 تأكيدات 4 ساعات:\n"

        f"• RSI: "
        f"{r['rsi']:.1f}\n"

        f"• RSI يتحسن: "
        f"{'نعم' if r['rsi_up'] else 'لا'}\n"

        f"• MACD إيجابي: "
        f"{'نعم' if r['macd_positive'] else 'لا'}\n"

        f"• MACD يتحسن/تقاطع: "
        f"{'نعم' if (r['macd_up'] or r['macd_cross']) else 'لا'}\n"

        f"• فوق MA20: "
        f"{'نعم' if r['ma20'] else 'لا'}\n"

        f"• فوق MA50: "
        f"{'نعم' if r['ma50'] else 'لا'}\n"

        f"• حجم 4H: "
        f"{r['volume_ratio']:.2f}x\n"

        f"• اختبارات الدعم: "
        f"{r['tests']}\n"

        f"• اختبارات ناجحة: "
        f"{r['successful']}\n"

        f"• تأكيد 15 دقيقة: "
        f"{'نعم' if r['small_confirmed'] else 'لا'}\n\n"

        "📌 عوامل إضافية:\n"

        f"• الفلوت: "
        f"{millions(r['float'])}\n"

        f"• البيع على المكشوف: "
        f"{number(r['short'])}\n"

        f"• Short أقل من 50 ألف: "
        f"{'نعم' if short_ok else 'لا'}\n\n"

        "📰 المحفزات الإيجابية:\n"

        f"{news}\n\n"

        f"⭐ نقاط الإشارة: "
        f"{r['score']}\n\n"

        "📍 طريقة الدخول:\n"

        "دخول تدريجي على دفعات بعد "
        "ثبات الدعم وإعادة الاختبار، "
        "وليس بكامل السيولة.\n\n"

        "⚠️ تنبيه آلي فني وليس توصية مالية. "
        "راجع الخبر والشارت وإدارة المخاطر قبل الدخول."
    )

    try:

        response = requests.post(

            (
                "https://api.telegram.org/"
                f"bot{TELEGRAM_TOKEN}/sendMessage"
            ),

            data={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": msg,
            },

            timeout=20,
        )

        if response.ok:

            print(
                f"✅ تم إرسال Telegram: "
                f"{r['ticker']}",
                flush=True,
            )

            return True

        print(
            "❌ Telegram:",
            response.text,
            flush=True,
        )

    except Exception as e:

        print(
            "❌ خطأ Telegram:",
            e,
            flush=True,
        )

    return False


# =========================
# تحليل السهم
# =========================

def analyze(
    ticker,
):

    try:

        raw = yf.download(

            ticker,

            period=DAILY_PERIOD,

            interval="1d",

            auto_adjust=False,

            progress=False,

            threads=False,
        )

        daily = clean_df(
            raw
        )

        if len(daily) < 60:
            return None

        current = num(
            daily["Close"].iloc[-1]
        )

        if (
            current is None
            or not (
                MIN_PRICE
                <= current
                <= MAX_PRICE
            )
        ):
            return None

        rally = detect_rally(
            daily
        )

        if rally is None:
            return None

        if (
            current
            > rally["high"] * 1.03
        ):
            return None

        support = support_level(
            daily,
            rally,
        )

        if (
            support is None
            or not support_not_broken(
                daily,
                rally,
                support,
            )
        ):
            return None

        tests = support_tests(
            daily,
            rally,
            support,
        )

        near = (
            support * SUPPORT_LOW
            <= current
            <= support * SUPPORT_HIGH
        )

        if (
            not near
            and not tests["second"]
        ):
            return None

        # =====================
        # 1H -> 4H
        # =====================

        one_h_raw = yf.download(

            ticker,

            period=ONE_HOUR_PERIOD,

            interval="1h",

            auto_adjust=False,

            progress=False,

            threads=False,
        )

        four_h = build_4h(
            one_h_raw
        )

        tech = analyze_4h(
            four_h
        )

        if not tech["ready"]:
            return None

        if (
            tech["rsi"] is None
            or tech["rsi"] > RSI_MAX
        ):
            return None

        momentum = (
            tech["rsi_up"]
            or tech["macd_up"]
            or tech["macd_cross"]
        )

        if not momentum:
            return None

        # =====================
        # المقاومات والأهداف
        # =====================

        levels = resistances(

            daily,

            current,

            rally["high"],
        )

        targets = make_targets(

            current,

            rally["low"],

            rally["high"],

            levels,
        )

        if not targets:
            return None

        # =====================
        # منطقة الدخول
        # =====================

        entry_low = (
            support * 0.98
        )

        entry_high = (
            support * 1.12
        )

        stop = (
            support
            * SUPPORT_BREAK
        )

        # =====================
        # Float / Short
        # =====================

        float_shares, short_shares = (
            get_float_short(
                ticker
            )
        )

        if float_shares is not None:

            if (
                float_shares
                < MIN_FLOAT
            ):
                return None

            if (
                float_shares
                > MAX_FLOAT
            ):
                return None

        # =====================
        # فريم 15 دقيقة
        # =====================

        small = confirm_15m(
            ticker
        )

        # =====================
        # الأخبار
        # =====================

        news = positive_news(
            ticker
        )

        # =====================
        # نظام النقاط
        # =====================

        score = 0

        reasons = []

        if tests["second"]:

            score += 2

            reasons.append(
                "اختباران للدعم"
            )

        if tests["stable"]:

            score += 1

            reasons.append(
                "الدعم مستقر"
            )

        if tech["rsi_up"]:

            score += 1

            reasons.append(
                "RSI يتحسن"
            )

        if tech["macd_positive"]:

            score += 1

            reasons.append(
                "MACD إيجابي"
            )

        if (
            tech["macd_up"]
            or tech["macd_cross"]
        ):

            score += 1

            reasons.append(
                "MACD يتحسن/تقاطع"
            )

        if tech["ma20"]:

            score += 1

            reasons.append(
                "فوق MA20"
            )

        if tech["ma50"]:

            score += 1

            reasons.append(
                "فوق MA50"
            )

        if tech["volume_ok"]:

            score += 1

            reasons.append(
                "حجم داعم"
            )

        if small["confirmed"]:

            score += 2

            reasons.append(
                "تأكيد 15 دقيقة"
            )

        if (
            short_shares is not None
            and short_shares
            < MAX_SHORT
        ):

            score += 1

            reasons.append(
                "Short منخفض"
            )

        if news:

            score += 1

            reasons.append(
                "محفز إيجابي"
            )

        if score < MIN_SCORE:
            return None

        # يجب وجود تأكيد زخم واضح
        if not (
            small["confirmed"]
            or tech["macd_cross"]
            or (
                tech["macd_positive"]
                and tech["rsi_up"]
            )
        ):
            return None

        return {

            "ticker": ticker,

            "price": current,

            "rally_low": rally["low"],

            "rally_high": rally["high"],

            "rally_pct": rally["percent"],

            "sessions": rally["sessions"],

            "support": support,

            "tests": tests["tests"],

            "successful": tests["successful"],

            "entry_low": entry_low,

            "entry_high": entry_high,

            "stop": stop,

            "targets": targets,

            "rsi": tech["rsi"],

            "rsi_up": tech["rsi_up"],

            "macd_positive": tech["macd_positive"],

            "macd_up": tech["macd_up"],

            "macd_cross": tech["macd_cross"],

            "ma20": tech["ma20"],

            "ma50": tech["ma50"],

            "volume_ratio": tech["volume_ratio"],

            "float": float_shares,

            "short": short_shares,

            "small_confirmed": small["confirmed"],

            "news": news,

            "score": score,

            "reasons": reasons,

            "rally_date": str(
                rally["high_date"]
            ),
        }

    except Exception as e:

        print(
            f"[{ticker}] خطأ: {e}",
            flush=True,
        )

        return None


# =========================
# فحص دفعة
# =========================

def scan_chunk(
    tickers,
    n,
    total,
):

    print(
        f"[دفعة {n}/{total}] "
        f"{len(tickers)} سهم",
        flush=True,
    )

    for ticker in tickers:

        result = analyze(
            ticker
        )

        if not result:
            continue

        print(

            f"⭐ مرشح {ticker} | "

            f"نقاط {result['score']} | "

            f"السعر "
            f"{price(result['price'])} | "

            f"الدعم "
            f"{price(result['support'])}",

            flush=True,
        )

        key = (

            ticker,

            result[
                "rally_date"
            ],

            round(
                result[
                    "entry_low
