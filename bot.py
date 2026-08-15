# -*- coding: utf-8 -*-

"""
PULLBACK RECOVERY RADAR - BASE BOTTOM VERSION
نسخة محسّنة للسرعة على GitHub Actions

الفكرة:
1) تحميل الأسهم على دفعات بدل سهم واحد كل مرة.
2) فلترة السعر والحركة السابقة أولاً.
3) الفحص العميق 1H و4H و15M فقط للمرشحين.
4) Telegram فقط عند اكتمال إشارة الدخول.
"""

import os
import warnings
import time
import requests
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime

warnings.filterwarnings("ignore")


# ============================================================
# Telegram
# ============================================================

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ============================================================
# الإعدادات
# ============================================================

MIN_PRICE = 1.0
MAX_PRICE = 5.0

MIN_RALLY = 100.0

MIN_SESSIONS = 4
MAX_SESSIONS = 20

MIN_AGE = 4
MAX_AGE = 31

SHORT_MAX = 20_000

MIN_FLOAT = 5_000_000
MAX_FLOAT = 500_000_000

SUPPORT_TOL = 0.12
BREAK_TOL = 0.03

RSI_LOW = 30.0
RSI_IMPROVE = 1.5

# عدد الأسهم في كل طلب Yahoo
BATCH_SIZE = 100

# انتظار بسيط بين دفعات Yahoo
BATCH_SLEEP = 0.5

# إعادة المحاولة عند فشل Yahoo
RETRIES = 2


# ============================================================
# أدوات مساعدة
# ============================================================

def n(x):
    try:
        x = float(x)

        if np.isnan(x) or np.isinf(x):
            return None

        return x

    except:
        return None


def p(x):
    x = n(x)

    if x is None:
        return "غير متوفر"

    return f"${x:.2f}"


def mil(x):
    x = n(x)

    if x is None:
        return "غير متوفر"

    return f"{x / 1e6:.1f} مليون"


def numfmt(x):
    x = n(x)

    if x is None:
        return "غير متوفر"

    return f"{x:,.0f}"


# ============================================================
# تنظيف البيانات
# ============================================================

def clean(df):

    if df is None or df.empty:
        return pd.DataFrame()

    d = df.copy()

    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)

    d.columns = [str(x) for x in d.columns]

    cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    if not all(x in d.columns for x in cols):
        return pd.DataFrame()

    return d.dropna(subset=cols)


# ============================================================
# RSI
# ============================================================

def rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    ag = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    al = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    return 100 - (
        100 /
        (
            1 +
            ag / al.replace(0, np.nan)
        )
    )


# ============================================================
# MACD
# ============================================================

def macd(close):

    a = close.ewm(
        span=12,
        adjust=False
    ).mean()

    b = close.ewm(
        span=26,
        adjust=False
    ).mean()

    m = a - b

    s = m.ewm(
        span=9,
        adjust=False
    ).mean()

    return m, s, m - s


# ============================================================
# 4 ساعات
# ============================================================

def four_h(one_h):

    d = clean(one_h)

    if d.empty:
        return d

    try:

        return (
            d.resample("4h")
            .agg({
                "Open": "first",
                "High": "max",
                "Low": "min",
                "Close": "last",
                "Volume": "sum"
            })
            .dropna()
        )

    except:

        return pd.DataFrame()


# ============================================================
# قائمة الأسهم
# ============================================================

def tickers():

    out = []

    sources = [
        (
            "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
            "Symbol"
        ),
        (
            "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
            "ACT Symbol"
        )
    ]

    for url, col in sources:

        try:

            d = pd.read_csv(
                url,
                sep="|"
            )

            d = d[d["Test Issue"] == "N"]

            out += (
                d[col]
                .dropna()
                .astype(str)
                .tolist()
            )

        except Exception as e:

            print(
                "قائمة الأسهم:",
                e,
                flush=True
            )

    result = sorted(
        set(
            x.strip().upper()
            for x in out
            if x.strip().isalpha()
            and len(x.strip()) <= 5
        )
    )

    return result


# ============================================================
# تحميل البيانات على دفعات
# ============================================================

def download_batch(symbols):

    if not symbols:
        return None

    for attempt in range(RETRIES + 1):

        try:

            data = yf.download(
                tickers=symbols,
                period="180d",
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=True,
                group_by="ticker"
            )

            if data is not None and not data.empty:
                return data

        except Exception as e:

            print(
                f"خطأ تحميل دفعة {attempt + 1}: {e}",
                flush=True
            )

            time.sleep(2)

    return None


# ============================================================
# استخراج بيانات سهم من Batch
# ============================================================

def extract_symbol(batch, ticker):

    if batch is None or batch.empty:
        return pd.DataFrame()

    try:

        if isinstance(batch.columns, pd.MultiIndex):

            level0 = batch.columns.get_level_values(0)

            if ticker in level0:

                d = batch[ticker]

                return clean(d)

            # أحياناً Yahoo يرجع ترتيب مختلف
            level1 = batch.columns.get_level_values(1)

            if ticker in level1:

                d = batch.xs(
                    ticker,
                    axis=1,
                    level=1
                )

                return clean(d)

        else:

            return clean(batch)

    except:

        return pd.DataFrame()


# ============================================================
# اكتشاف الصعود السابق
# ============================================================

def detect_rally(d):

    d = clean(d)

    if len(d) < 70:
        return None

    today = datetime.now().date()

    best = None

    start_search = max(
        20,
        len(d) - 55
    )

    for end in range(
        start_search,
        len(d)
    ):

        hi = n(d.iloc[end].High)

        if hi is None:
            continue

        hdate = pd.Timestamp(
            d.index[end]
        ).date()

        age = (
            today -
            hdate
        ).days

        if not (
            MIN_AGE <= age <= MAX_AGE
        ):
            continue

        for sessions in range(
            MIN_SESSIONS,
            MAX_SESSIONS + 1
        ):

            start = end - sessions

            if start < 6:
                continue

            bw = d.iloc[
                max(0, start - 5):
                start + 2
            ]

            base = n(
                bw.Low.min()
            )

            if base is None or base <= 0:
                continue

            pct = (
                (hi - base) /
                base *
                100
            )

            if pct < MIN_RALLY:
                continue

            med = n(
                bw.Low.median()
            )

            if med is None:
                continue

            if abs(base - med) / base > 0.20:
                continue

            cand = {
                "base": base,
                "high": hi,
                "pct": pct,
                "sessions": sessions,
                "high_date": hdate,
                "start_date":
                    pd.Timestamp(
                        d.index[start]
                    ).date(),
                "start_pos": start,
                "end_pos": end
            }

            if (
                best is None
                or
                (
                    cand["high_date"],
                    cand["pct"]
                )
                >
                (
                    best["high_date"],
                    best["pct"]
                )
            ):

                best = cand

    return best


# ============================================================
# اختبار الدعم الأصلي
# ============================================================

def support_check(
    d,
    h,
    base,
    rally
):

    low = base * (
        1 - SUPPORT_TOL
    )

    high = base * (
        1 + SUPPORT_TOL
    )

    after_daily = d[
        d.index.date >= rally["high_date"]
    ]

    if after_daily.empty:
        return False, 0, False

    lowest = n(
        after_daily.Low.min()
    )

    if (
        lowest is not None
        and
        lowest <
        base * (1 - BREAK_TOL)
    ):

        return False, 0, False

    x = four_h(h)

    x = x[
        x.index.date >= rally["high_date"]
    ].tail(60)

    if x.empty:
        return False, 0, False

    tests = []

    for idx, row in x.iterrows():

        lo = n(row.Low)

        cl = n(row.Close)

        if lo is None or cl is None:
            continue

        if low <= lo <= high:

            tests.append(
                (
                    idx,
                    lo,
                    cl
                )
            )

    if not tests:
        return False, 0, False

    rejection = False

    for idx, lo, cl in tests[-10:]:

        rebound_from_low = (
            cl >= lo * 1.02
        )

        held_original_base = (
            cl >= base * 0.98
        )

        if (
            rebound_from_low
            and
            held_original_base
        ):

            rejection = True

    current = n(
        x.Close.iloc[-1]
    )

    near = (
        current is not None
        and
        low <= current <= high
    )

    return (
        near,
        len(tests),
        rejection
    )


# ============================================================
# التحليل الفني 4H
# ============================================================

def tech(h):

    x = four_h(h)

    if len(x) < 55:
        return None

    x["RSI"] = rsi(
        x.Close
    )

    (
        x["M"],
        x["S"],
        x["H"]
    ) = macd(
        x.Close
    )

    x["MA20"] = (
        x.Close.rolling(20).mean()
    )

    x["MA50"] = (
        x.Close.rolling(50).mean()
    )

    rr = (
        x.RSI
        .tail(3)
        .dropna()
    )

    latest = x.iloc[-1]

    if rr.empty:
        return None

    m = n(latest.M)
    s = n(latest.S)
    hist = n(latest.H)

    prevh = n(
        x.H.iloc[-2]
    )

    r = n(
        latest.RSI
    )

    prevr = n(
        x.RSI.iloc[-2]
    )

    v = n(
        latest.Volume
    )

    va = n(
        x.Volume.tail(20).mean()
    )

    return {

        "rsi": r,

        "oversold":
            rr.min() < RSI_LOW,

        "rsi_up":
            (
                r is not None
                and
                prevr is not None
                and
                r - prevr >= RSI_IMPROVE
            ),

        "macd_pos":
            (
                m is not None
                and
                s is not None
                and
                m > s
            ),

        "macd_up":
            (
                hist is not None
                and
                prevh is not None
                and
                hist > prevh
            ),

        "cross":
            (
                m is not None
                and
                s is not None
                and
                n(x.M.iloc[-2]) is not None
                and
                n(x.S.iloc[-2]) is not None
                and
                m > s
                and
                n(x.M.iloc[-2])
                <=
                n(x.S.iloc[-2])
            ),

        "ma20":
            n(latest.MA20),

        "ma50":
            n(latest.MA50),

        "above20":
            (
                n(latest.MA20)
                is not None
                and
                latest.Close
                >
                latest.MA20
            ),

        "above50":
            (
                n(latest.MA50)
                is not None
                and
                latest.Close
                >
                latest.MA50
            ),

        "volratio":
            (
                v / va
                if
                v is not None
                and
                va
                else 0
            )
    }


# ============================================================
# تأكيد 15 دقيقة
# ============================================================

def confirm15(ticker):

    try:

        d = clean(
            yf.download(
                ticker,
                period="10d",
                interval="15m",
                auto_adjust=False,
                progress=False,
                threads=False
            )
        )

        if len(d) < 40:
            return False

        d["RSI"] = rsi(
            d.Close
        )

        (
            d["M"],
            d["S"],
            _
        ) = macd(
            d.Close
        )

        a = d.iloc[-1]
        b = d.iloc[-2]

        r = n(a.RSI)
        rp = n(b.RSI)

        m = n(a.M)
        s = n(a.S)

        v = n(a.Volume)

        av = n(
            d.Volume.tail(20).mean()
        )

        return (
            sum([
                (
                    r is not None
                    and
                    rp is not None
                    and
                    r > rp
                ),

                (
                    m is not None
                    and
                    s is not None
                    and
                    m > s
                ),

                (
                    v is not None
                    and
                    av
                    and
                    v >= av * 1.05
                )
            ])
            >= 2
        )

    except:

        return False


# ============================================================
# Float / Short
# ============================================================

def float_short(t):

    try:

        info = yf.Ticker(t).info

        return (
            n(info.get("floatShares")),
            n(info.get("sharesShort"))
        )

    except:

        return None, None


# ============================================================
# الأخبار
# ============================================================

def news(t):

    keys = [
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
        "deal"
    ]

    out = []

    try:

        items = (
            yf.Ticker(t).news
            or []
        )

        for z in items[:10]:

            c = z.get(
                "content",
                {}
            )

            if isinstance(c, dict):

                title = c.get(
                    "title",
                    ""
                )

            else:

                title = z.get(
                    "title",
                    ""
                )

            if any(
                k in str(title).lower()
                for k in keys
            ):

                out.append(
                    str(title)
                )

    except:

        pass

    return out[:3]


# ============================================================
# الأهداف
# ============================================================

def targets(
    d,
    base,
    hi,
    current
):

    levels = []

    for z in (
        d.High.tail(100)
        .dropna()
    ):

        z = n(z)

        if (
            z
            and
            z > current * 1.05
            and
            z < current * 4
            and
            not any(
                abs(z - a) / a < 0.04
                for a in levels
            )
        ):

            levels.append(z)

    if (
        hi > current * 1.05
        and
        not any(
            abs(hi - a) / a < 0.04
            for a in levels
        )
    ):

        levels.append(hi)

    diff = hi - base

    for z in [
        hi + diff * 0.272,
        hi + diff * 0.618,
        hi + diff,
        hi + diff * 1.618
    ]:

        if (
            z > current * 1.10
            and
            not any(
                abs(z - a) / a < 0.04
                for a in levels
            )
        ):

            levels.append(z)

    return sorted(levels)[:5]


# ============================================================
# Telegram Alert
# ============================================================

def alert(r):

    if (
        not TELEGRAM_TOKEN
        or
        not TELEGRAM_CHAT_ID
    ):

        return False

    ts = "\n".join(
        f"🎯 الهدف {i}: {p(x)}"
        for i, x in enumerate(
            r["targets"],
            1
        )
    )

    ns = (
        "\n".join(
            "• " + x
            for x in r["news"]
        )
        if r["news"]
        else
        "لا يوجد محفز إيجابي واضح في البيانات المتاحة."
    )

    msg = (
        "🚨 إشارة دخول من القاع 🚨\n\n"

        f"📌 السهم: {r['ticker']}\n"
        f"💵 السعر الحالي: {p(r['price'])}\n\n"

        "📈 الحركة السابقة:\n"
        f"• الصعود: +{r['pct']:.1f}%\n"
        f"• مدة الصعود: {r['sessions']} جلسات\n"
        f"• قاع بداية الصعود: {p(r['base'])}\n"
        f"• قمة الصعود: {p(r['high'])}\n\n"

        "🟢 منطقة الدخول حول القاع الأصلي (مرنة):\n"
        f"• من {p(r['entry_low'])} إلى {p(r['entry_high'])}\n"
        f"• الدعم الأصلي: {p(r['base'])}\n"
        f"• وقف الخسارة: {p(r['stop'])}\n\n"

        f"{ts}\n\n"

        "📊 تأكيدات 4 ساعات:\n"
        f"• RSI: {r['rsi']:.1f}\n"
        "• RSI تحت 30 مؤخرًا: نعم\n"
        "• RSI يتحسن: نعم\n"
        f"• MACD إيجابي: {'نعم' if r['macd_pos'] else 'لا'}\n"
        f"• MACD يتحسن: {'نعم' if r['macd_up'] else 'لا'}\n"
        f"• تقاطع MACD: {'نعم' if r['cross'] else 'لا'}\n"
        f"• فوق MA20: {'نعم' if r['above20'] else 'لا'}\n"
        f"• فوق MA50: {'نعم' if r['above50'] else 'لا'}\n"
        f"• حجم 4H: {r['volratio']:.2f}x\n\n"

        "🛡️ الدعم:\n"
        f"• اختبارات الدعم: {r['tests']}\n"
        "• رفض كسر الدعم: نعم\n"
        "• تأكيد 15 دقيقة: نعم\n\n"

        "📌 عوامل إضافية:\n"
        f"• الفلوت: {mil(r['float'])}\n"
        f"• الشورت: {numfmt(r['short'])}\n"
        "• الشورت ≤20 ألف: نعم\n\n"

        "📰 محفزات إيجابية:\n"
        f"{ns}\n\n"

        "📍 الدخول على دفعات حول الدعم الأصلي، "
        "وليس مطاردة السهم بعد ابتعاده عن القاع.\n\n"

        "⚠️ تنبيه آلي فني وليس ضمانًا للربح. "
        "راجع الخبر والشارت وإدارة المخاطر."
    )

    try:

        url = (
            f"https://api.telegram.org/"
            f"bot{TELEGRAM_TOKEN}/sendMessage"
        )

        q = requests.post(
            url,
            data={
                "chat_id":
                    TELEGRAM_CHAT_ID,
                "text":
                    msg
            },
            timeout=20
        )

        print(
            "Telegram:",
            q.text,
            flush=True
        )

        return q.ok

    except Exception as e:

        print(
            "Telegram error:",
            e,
            flush=True
        )

        return False


# ============================================================
# التحليل العميق
# ============================================================

def analyze(t, d):

    try:

        d = clean(d)

        if len(d) < 70:
            return None

        current = n(
            d.Close.iloc[-1]
        )

        if (
            current is None
            or
            not (
                MIN_PRICE
                <= current
                <= MAX_PRICE
            )
        ):

            return None

        # ----------------------------------------------------
        # الصعود السابق
        # ----------------------------------------------------

        r = detect_rally(d)

        if not r:
            return None

        # ----------------------------------------------------
        # السعر الآن قريب من قاعدة الصعود
        # ----------------------------------------------------

        entry_low = (
            r["base"] * 0.97
        )

        entry_high = (
            r["base"] * 1.12
        )

        if not (
            entry_low
            <= current
            <= entry_high
        ):

            return None

        # ----------------------------------------------------
        # تحميل 1H فقط للمرشحين
        # ----------------------------------------------------

        h = clean(
            yf.download(
                t,
                period="30d",
                interval="1h",
                auto_adjust=False,
                progress=False,
                threads=False
            )
        )

        if len(h) < 150:
            return None

        # ----------------------------------------------------
        # الدعم
        # ----------------------------------------------------

        (
            near,
            tests,
            rejection
        ) = support_check(
            d,
            h,
            r["base"],
            r
        )

        if (
            not near
            or
            tests < 1
            or
            not rejection
        ):

            return None

        # ----------------------------------------------------
        # 4H
        # ----------------------------------------------------

        tx = tech(h)

        if not tx:
            return None

        if not tx["oversold"]:
            return None

        if not tx["rsi_up"]:
            return None

        # ----------------------------------------------------
        # Float / Short
        # ----------------------------------------------------

        fl, sh = float_short(t)

        if (
            sh is None
            or
            sh > SHORT_MAX
        ):

            return None

        if (
            fl is not None
            and
            not (
                MIN_FLOAT
                <= fl
                <= MAX_FLOAT
            )
        ):

            return None

        # ----------------------------------------------------
        # 15 دقيقة
        # ----------------------------------------------------

        if not confirm15(t):
            return None

        # ----------------------------------------------------
        # Score
        # ----------------------------------------------------

        score = (
            2
            + 2
            + 2
            + (
                1
                if tx["macd_up"]
                else 0
            )
            + (
                1
                if tx["macd_pos"]
                else 0
            )
            + (
                1
                if tx["cross"]
                else 0
            )
            + (
                1
                if (
                    tx["above20"]
                    or
                    tx["above50"]
                )
                else 0
            )
            + (
                1
                if tx["volratio"] >= 1.05
                else 0
            )
        )

        if score < 6:
            return None

        # ----------------------------------------------------
        # الأهداف
        # ----------------------------------------------------

        tg = targets(
            d,
            r["base"],
            r["high"],
            current
        )

        if not tg:
            return None

        # ----------------------------------------------------
        # النتيجة
        # ----------------------------------------------------

        return {

            "ticker":
                t,

            "price":
                current,

            "base":
                r["base"],

            "high":
                r["high"],

            "pct":
                r["pct"],

            "sessions":
                r["sessions"],

            "entry_low":
                entry_low,

            "entry_high":
                entry_high,

            "stop":
                r["base"]
                * (1 - BREAK_TOL),

            "rsi":
                tx["rsi"],

            "macd_pos":
                tx["macd_pos"],

            "macd_up":
                tx["macd_up"],

            "cross":
                tx["cross"],

            "above20":
                tx["above20"],

            "above50":
                tx["above50"],

            "volratio":
                tx["volratio"],

            "tests":
                tests,

            "float":
                fl,

            "short":
                sh,

            "targets":
                tg,

            "news":
                news(t)
        }

    except Exception as e:

        print(
            f"[{t}] {e}",
            flush=True
        )

        return None


# ============================================================
# الفحص الرئيسي
# ============================================================

ALERTED = set()


def scan():

    print(
        "🚀 فحص اصطياد القاع الأصلي...",
        flush=True
    )

    ts = tickers()

    print(
        f"تم تحميل {len(ts)} سهم",
        flush=True
    )

    found = 0
    candidate_count = 0

    total_batches = (
        len(ts) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    # ========================================================
    # المرحلة الأولى:
    # تحميل البيانات اليومية على دفعات
    # ========================================================

    for batch_number, start in enumerate(
        range(
            0,
            len(ts),
            BATCH_SIZE
        ),
        1
    ):

        batch_symbols = ts[
            start:
            start + BATCH_SIZE
        ]

        print(
            f"📦 الدفعة "
            f"{batch_number}/{total_batches} "
            f"| {len(batch_symbols)} سهم",
            flush=True
        )

        data = download_batch(
            batch_symbols
        )

        if data is None:

            print(
                "⚠️ فشل تحميل هذه الدفعة، ننتقل للدفعة التالية.",
                flush=True
            )

            continue

        # ====================================================
        # فلترة أولية سريعة
        # ====================================================

        for t in batch_symbols:

            d = extract_symbol(
                data,
                t
            )

            if d.empty:
                continue

            # السعر الحالي
            current = n(
                d.Close.iloc[-1]
            )

            if (
                current is None
                or
                not (
                    MIN_PRICE
                    <= current
                    <= MAX_PRICE
                )
            ):

                continue

            # الصعود السابق
            rally = detect_rally(d)

            if not rally:
                continue

            # السعر قريب من القاع الأصلي
            entry_low = (
                rally["base"] * 0.97
            )

            entry_high = (
                rally["base"] * 1.12
            )

            if not (
                entry_low
                <= current
                <= entry_high
            ):

                continue

            candidate_count += 1

            print(
                f"🎯 مرشح أولي: {t} "
                f"| السعر {p(current)} "
                f"| الدعم {p(rally['base'])} "
                f"| الصعود +{rally['pct']:.1f}%",
                flush=True
            )

            # =================================================
            # الفحص العميق
            # =================================================

            result = analyze(
                t,
                d
            )

            if result:

                found += 1

                key = (
                    t,
                    round(
                        result["base"],
                        4
                    ),
                    round(
                        result["high"],
                        4
                    )
                )

                print(
                    f"🔥 إشارة مكتملة: {t} "
                    f"| السعر {p(result['price'])} "
                    f"| الدعم الأصلي {p(result['base'])} "
                    f"| +{result['pct']:.1f}%",
                    flush=True
                )

                if key not in ALERTED:

                    if alert(result):

                        ALERTED.add(key)

        # انتظار بسيط
        time.sleep(
            BATCH_SLEEP
        )

    # ========================================================
    # النهاية
    # ========================================================

    print(
        "========================================",
        flush=True
    )

    print(
        f"انتهى الفحص.",
        flush=True
    )

    print(
        f"المرشحات الأولية: {candidate_count}",
        flush=True
    )

    print(
        f"إشارات الدخول المكتملة: {found}",
        flush=True
    )

    print(
        "========================================",
        flush=True
    )


# ============================================================
# التشغيل
# ============================================================

if __name__ == "__main__":

    scan()
