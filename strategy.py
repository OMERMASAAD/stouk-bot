# -*- coding: utf-8 -*-
"""
رادار القاع — استراتيجية «الرجل الثانية» (Bottom Radar / Second Leg Setup)
البيانات: Yahoo Finance (شموع يومية + شموع 1h/4h لتأكيد الأنماط + أخبار وتقويم).

الفكرة:
  سهم منخفض الـ Float ($1–$5) صعد صعودًا هائلًا (+100% أو أكثر خلال آخر 20 جلسة)
  ثم عاد إلى قاعدة الانطلاق (Base Support) خلال مدة لا تتجاوز 20 جلسة من القمة،
  وتكوّن/يثبت عند القاع بفوليوم ضعيف، ويبدأ بتشكيل Pivot مبكر للرجل الثانية.

الفلاتر الصارمة (تُستبعد في حال الإخلال بأي منها):
  1) السعر الحالي بين $1 و$5.
  2) صعود سابق: (أعلى 20 جلسة - أدنى 20 جلسة) / أدنى 20 جلسة >= 1.00 (+100%).
  3) Float <= 10M، ويُستبعد السهم إذا كان Float غير متوفر.
  4) Days_Since_Peak > 20 جلسة => استبعاد فوري (انتهت صلاحية الـ setup).

درجة الجاهزية (100 نقطة): قرب القاع 20 · نمط القاع 15 · ثبات الدعم 15 · RSI 10 ·
الفوليوم 10 · EMA20 10 · VWAP 5 · MACD 5 · محفز مستقبلي 10.

المراحل: 30–50 قيد المتابعة · 55–75 شبه جاهز · 80–100 جاهز فنيًا.
خطة الصفقة تُولد تلقائيًا عند «جاهز فنيًا» (الدخول = الإغلاق، الوقف = القاع -2%،
الهدف الأول = آخر Lower High أو فيبوناتشي 38.2%، الهدف الثاني = قمة الصعود).

الواجهة عربية بالكامل (نصوص/تسميات)، والكود والمفاتيح JSON بالإنجليزية.
"""
import numpy as np
import pandas as pd

from news import summarize_news

# ---------------- الإعدادات (الفلاتر الصارمة) ----------------
MIN_PRICE, MAX_PRICE = 1.0, 5.0            # فلتر السعر الحالي
RALLY_WINDOW = 20                          # نافذة قياس الصعود السابق (جلسة)
MIN_PRIOR_RALLY_PCT = 100.0                # الصعود السابق المطلوب (high-low)/low >= 1.00
MAX_FLOAT = 10_000_000                     # Float <= 10M وإلا استبعاد
MAX_DAYS_SINCE_PEAK = 20                   # انتهاء صلاحية الـ setup بعد 20 جلسة من القمة

# ---------------- إعدادات الأنماط والمناطق ----------------
NEAR_BASE = 0.15                           # «قريب من القاع» = داخل 15% فوق القاع
DIST_TOO_FAR = 0.20                        # أكثر من 20% فوق القاع => صفر نقاط (لا يُوصف بالقرب)
ZONE_TOL = 0.10                            # منطقة القاع = القاع ± 10%
BREAK_HARD = 0.15                          # كسر قوي تحت القاع => إلغاء الـ setup
MIN_CONSOLIDATION_BARS = 2                 # ثبات الدعم لجلستين متتاليتين على الأقل
RVOL_DRY = 0.80                            # جفاف الفوليوم (RVOL <= 0.8x)
RVOL_UPTICK = (1.00, 1.20)                 # بداية زخم خفيف (1.0x – 1.2x)
TIGHT_BODY_MAX = 0.06                      # الشمعة الضيقة: جسم <= 6%
TIGHT_RANGE_MAX = 0.09                     # ومدى الشمعة <= 9%
DOUBLE_BOTTOM_TOL = 0.05                   # تقارب القاعين في القاع المزدوج <= 5%
PATTERN_BOUNCE_MIN = 0.03                  # ارتداد أدنى بين القاعين (3%)
PATTERN_LOOKBACK = 60                      # نافذة البحث عن الأنماط (شموع الفريم الأصغر)

# ---------------- إعدادات خطة الصفقة ----------------
STOP_BUFFER = 0.02                         # الوقف = القاع -2%
FIB_RETRACE = 0.382                        # الهدف الأول البديل: فيبوناتشي 38.2%

# ---------------- نافذة المحفزات ----------------
CATALYST_MIN_DAYS, CATALYST_MAX_DAYS = 7, 30
ANALYSIS_DAYS = 30

STAGE_WATCH = "قيد المتابعة"
STAGE_SEMI = "شبه جاهز"
STAGE_READY = "جاهز فنيًا"
STAGE_ORDER = {STAGE_READY: 3, STAGE_SEMI: 2, STAGE_WATCH: 1}
STAGE_BANDS = {STAGE_WATCH: (30, 50), STAGE_SEMI: (55, 75), STAGE_READY: (80, 100)}

PATTERN_DOUBLE = "قاع مزدوج (Double Bottom)"
PATTERN_HIGHER_LOW = "قاع أعلى من قاع (Higher Low)"
PATTERN_NONE = "لا يوجد نمط قاع واضح بعد"

# نقاط الجاهزية (المجموع 100)
POINTS = {
    "near_base": 20, "pattern": 15, "stable": 15, "rsi": 10, "volume": 10,
    "ema20": 10, "vwap": 5, "macd": 5, "catalyst": 10,
}

WARNING_TAG = "🚨 تحذير: خافض لقيمة الأسهم / أزمة"

MISSING_LABELS = {
    "near_base": "السعر بعيد عن منطقة القاع (أكثر من 15%)",
    "pattern": "لم يتكوّن نمط قاع مزدوج أو قاع أعلى من قاع",
    "stable": "الأسعار لم تثبت فوق الدعم لجلستين متتاليتين بعد",
    "rsi": "RSI لم يبدأ التحسن من منطقة التشبع البيعي بعد",
    "volume": "الفوليوم لم يجف بعد ولا يوجد زخم خفيف مبكر",
    "ema20": "السعر لم يستعد EMA20 بعد",
    "vwap": "السعر لم يستعد VWAP المرتكز من قاع الثبات بعد",
    "macd": "MACD Histogram لم يتحسن بعد",
    "catalyst": "لا يوجد محفز مستقبلي معروف خلال 7–30 يومًا",
}

AR_MONTHS = {
    1: "يناير", 2: "فبراير", 3: "مارس", 4: "أبريل", 5: "مايو", 6: "يونيو",
    7: "يوليو", 8: "أغسطس", 9: "سبتمبر", 10: "أكتوبر", 11: "نوفمبر", 12: "ديسمبر",
}


# ---------------- المؤشرات ----------------
def rsi(series, period=14):
    """
    RSI بطريقة Wilder القياسية (SMA كبذرة أولية ثم تنعيم تكراري) — مطابق لما يظهر
    في TradingView/Yahoo. الطريقة السابقة (ewm بلا بذرة) كانت تعطي قيمًا خاطئة.
    """
    x = np.asarray(pd.Series(series).astype(float), dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if n < period + 1 or np.isnan(x[:period + 1]).any():
        return pd.Series(out, index=getattr(series, "index", None))
    delta = np.diff(x)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = float(gain[:period].mean())
    al = float(loss[:period].mean())
    out[period] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(period, n - 1):
        ag = (ag * (period - 1) + gain[i]) / period
        al = (al * (period - 1) + loss[i]) / period
        out[i + 1] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return pd.Series(out, index=getattr(series, "index", None))


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def macd(series):
    fast, slow = ema(series, 12), ema(series, 26)
    line = fast - slow
    signal = ema(line, 9)
    return line, signal, line - signal


def rvol_series(volume, window=20):
    """RVOL = حجم الجلسة / متوسط حجم آخر 20 جلسة (بدون الجلسة الحالية)."""
    base = volume.shift(1).rolling(window).mean()
    return volume / base.replace(0, np.nan)


def anchored_vwap(d, start_abs):
    x = d.iloc[max(0, int(start_abs)):]
    if x.empty:
        return float(d["Close"].iloc[-1])
    tp = (x["High"].astype(float) + x["Low"].astype(float) + x["Close"].astype(float)) / 3
    vol = x["Volume"].astype(float)
    tot = float(vol.sum())
    return float((tp * vol).sum() / tot) if tot > 0 else float(x["Close"].iloc[-1])


def technicals(d, n=None):
    """مؤشرات فنية على الشموع الممررة (n=None: كلها)."""
    x = d if n is None else d.tail(n)
    close = x["Close"].astype(float)
    open_ = x["Open"].astype(float)
    volume = x["Volume"].astype(float)
    rrsi = rsi(close)
    e9, e20, e50 = ema(close, 9), ema(close, 20), ema(close, 50)
    ml, ms, mh = macd(close)
    rv = rvol_series(volume)

    r, r_prev = float(rrsi.iloc[-1]), float(rrsi.iloc[-2])
    mh_now, mh_prev = float(mh.iloc[-1]), float(mh.iloc[-2])
    last_rv = float(rv.iloc[-1]) if pd.notna(rv.iloc[-1]) else 0.0
    body = abs(float(close.iloc[-1]) - float(open_.iloc[-1])) / float(open_.iloc[-1]) if open_.iloc[-1] else 0.0

    ema20_reclaim = bool(close.iloc[-1] > e20.iloc[-1])
    ema20_rising = bool(e20.iloc[-1] > e20.iloc[-4]) if len(e20) >= 4 else False
    macd_improving = bool(mh_now > mh_prev)

    positive = sum([
        bool(r > r_prev), macd_improving,
        bool(ema20_reclaim and ema20_rising),
        bool(last_rv >= 1.2), bool(close.iloc[-1] > e50.iloc[-1]),
    ])
    return {
        "rsi": round(r, 2), "rsi_prev": round(r_prev, 2),
        "rsi_oversold_recent": bool(float(rrsi.min()) < 30),
        "rsi_turning_up": bool(r > r_prev),
        "ema9": round(float(e9.iloc[-1]), 4),
        "ema20": round(float(e20.iloc[-1]), 4), "ema50": round(float(e50.iloc[-1]), 4),
        "ema20_reclaim": ema20_reclaim, "ema20_rising": ema20_rising,
        "ema50_reclaim": bool(close.iloc[-1] > e50.iloc[-1]),
        "macd": round(float(ml.iloc[-1]), 5), "macd_signal": round(float(ms.iloc[-1]), 5),
        "macd_hist": round(mh_now, 5), "macd_hist_prev": round(mh_prev, 5),
        "macd_cross": bool(ml.iloc[-1] > ms.iloc[-1] and ml.iloc[-2] <= ms.iloc[-2]),
        "macd_improving": macd_improving,
        "rvol": round(last_rv, 2), "volume_ratio": round(last_rv, 2),
        "body_pct": round(body * 100, 2),
        "volume_confirm": bool(last_rv >= 1.2),
        "positive_confirmations": int(positive),
    }


# ---------------- اكتشاف حدث الصعود (Pump) ----------------
def find_surge_event(d):
    """
    يبحث عن أحدث قمة صعود تحقق الشرطين معًا في نافذة 20 جلسة المنتهية عند القمة:
      - الصعود: (أعلى - أدنى) / أدنى >= +100%
      - القمة = أعلى سعر داخل النافذة (قمة حقيقية وليست قمة داخلية)
    يعيد dict يحتوي event_idx (موقع القمة) و base (قاع الانطلاق) و high (القمة).
    """
    if d is None or len(d) < RALLY_WINDOW + 2:
        return None
    n = len(d)
    highs = d["High"].astype(float).values
    lows = d["Low"].astype(float).values
    # نبحث من الأحدث للأقدم: أحدث قمة مؤهلة هي المرجع لاحتساب Days_Since_Peak
    for p in range(n - 1, RALLY_WINDOW - 1, -1):
        s = p - RALLY_WINDOW + 1
        window_high = float(highs[s:p + 1].max())
        if highs[p] < window_high:
            continue                       # ليست أعلى قمة في النافذة
        b = s + int(np.argmin(lows[s:p + 1]))
        base = float(lows[b])
        if base <= 0:
            continue
        rally = (float(highs[p]) - base) / base * 100
        if rally < MIN_PRIOR_RALLY_PCT:
            continue
        return {
            "event_idx": int(p),
            "event_date": str(d.index[p].date()),
            "event_price": float(d["Close"].iloc[p]),
            "base": base, "base_idx": int(b), "base_date": str(d.index[b].date()),
            "high": float(highs[p]), "high_idx": int(p),
            "rally_pct": float(rally),
        }
    return None


# ---------------- أدوات الهيكل السعري ----------------
def _pivot_highs(highs, start, end):
    """مؤشرات القمم المحلية بين start و end (شاملة)."""
    out = []
    for i in range(max(1, start), min(end, len(highs) - 2) + 1):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
            out.append(i)
    return out


def _swing_lows(lows, k=1):
    """قيعان محلية (Swing Lows) بمقارنة k شموع على كل جانب."""
    return [i for i in range(k, len(lows) - k)
            if lows[i] <= min(lows[i - k:i]) and lows[i] <= min(lows[i + 1:i + k + 1])]


def last_lower_high(d, start_abs, end_abs):
    """آخر Lower High (أعلى قمة محلية) بين القمة والقاع — يُستخدم كهدف أول."""
    highs = d["High"].astype(float).values
    if end_abs <= start_abs:
        return None
    piv = _pivot_highs(highs, start_abs, end_abs - 1)
    if piv:
        return float(highs[piv[-1]])
    seg = highs[start_abs:end_abs]
    return float(seg.max()) if len(seg) else None


# ---------------- أنماط القاع (1h / 4h) ----------------
def detect_base_pattern(f, base):
    """
    يكتشف نمط القاع على فريم أقصر (1h أو 4h) مع الثبات فوق القاع:
      - قاع مزدوج: قاعان متقاربان (<= 5%) يفصلهما ارتداد >= 3%.
      - قاع أعلى من قاع: آخر قاعين صاعدان مع ثبات الأول فوق القاع.
    يعيد dict {"type", "detail", "level"} أو None.
    """
    if f is None or len(f) < 12:
        return None
    x = f.tail(PATTERN_LOOKBACK)
    lows = x["Low"].astype(float).values
    highs = x["High"].astype(float).values
    if len(lows) < 12:
        return None
    floor = float(base) * (1 - ZONE_TOL)      # الحد الأدنى المقبول: القاع -10%
    sw = [i for i in _swing_lows(lows, 1) if lows[i] >= floor]
    if not sw:
        return None

    # ---- قاع مزدوج ----
    for a, b in zip(sw, sw[1:]):
        if b - a < 2:
            continue
        l1, l2 = float(lows[a]), float(lows[b])
        if l1 <= 0 or abs(l1 - l2) / min(l1, l2) > DOUBLE_BOTTOM_TOL:
            continue
        neck = float(highs[a:b + 1].max())
        if neck < max(l1, l2) * (1 + PATTERN_BOUNCE_MIN):
            continue
        return {
            "type": PATTERN_DOUBLE,
            "detail": f"قاعان عند {min(l1, l2):.2f} و{max(l1, l2):.2f} (فارق أقل من 5%)",
            "level": round(max(l1, l2), 4),
            "neckline": round(neck, 4),
            "bars": int(b - a),
        }

    # ---- قاع أعلى من قاع ----
    if len(sw) >= 2:
        a, b = sw[-2], sw[-1]
        l1, l2 = float(lows[a]), float(lows[b])
        if b > a and l2 > l1 * 1.005 and l1 >= floor:
            return {
                "type": PATTERN_HIGHER_LOW,
                "detail": f"قاع أعلى: {l1:.2f} ثم {l2:.2f}",
                "level": round(l2, 4),
                "neckline": round(float(highs[b:].max()), 4),
                "bars": int(b - a),
            }
    return None


# ---------------- تقييم الجلسات بعد القمة ----------------
def _post_peak_state(d, event):
    """قاع ما بعد القمة، جلوس الثبات، الهبوط الأقصى، وكسر الدعم."""
    n = len(d)
    ev = int(event["event_idx"])
    base = float(event["base"])
    highs = d["High"].astype(float).values
    lows = d["Low"].astype(float).values
    closes = d["Close"].astype(float).values
    since_lows = lows[ev + 1:]
    m = len(since_lows)
    trough_rel = int(np.argmin(since_lows))
    trough_abs = ev + 1 + trough_rel
    trough_low = float(since_lows[trough_rel])
    bars_since_trough = m - 1 - trough_rel

    zone_low = base * (1 - ZONE_TOL)
    holding = all(lows[i] >= zone_low for i in range(trough_abs, n))
    holding = bool(holding and closes[-1] >= zone_low)
    stable = bool(bars_since_trough >= MIN_CONSOLIDATION_BARS and holding and lows[-1] > trough_low)

    max_dd = (float(event["high"]) - trough_low) / float(event["high"]) * 100 if event["high"] else 0.0
    below_zone = closes[-1] < zone_low
    deep_break = trough_low < base * (1 - BREAK_HARD)
    new_lows = bars_since_trough <= 0
    real_breakdown = bool(below_zone and (deep_break or new_lows))
    false_breakdown = bool(trough_low < zone_low and not below_zone)
    return {
        "trough_abs": trough_abs, "trough_low": trough_low,
        "bars_since_trough": int(bars_since_trough),
        "holding": holding, "stable": stable,
        "max_drawdown_pct": max_dd,
        "real_breakdown": real_breakdown, "false_breakdown": false_breakdown,
        "lower_high": last_lower_high(d, ev + 1, trough_abs),
    }


def _entry_plan(d, event, state, ready):
    """خطة الصفقة التلقائية (تُعرض عند «جاهز فنيًا»)."""
    price = float(d["Close"].iloc[-1])
    base = float(event["base"])
    peak = float(event["high"])
    trough_low = float(state["trough_low"])

    # الوقف: القاع (Base Support) - %2، ولا يزيد عن 1% تحت السعر الحالي
    stop = base * (1 - STOP_BUFFER)
    stop = min(stop, trough_low * (1 - STOP_BUFFER))
    stop = min(stop, price * 0.99)

    # الهدف الأول: آخر Lower High أو فيبوناتشي 38.2% من الهبوط
    fib = trough_low + FIB_RETRACE * (peak - trough_low)
    lh = state.get("lower_high")
    tp1 = float(lh) if lh and float(lh) > price * 1.01 else float(fib)
    if tp1 <= price:
        tp1 = float(fib) if fib > price else float(peak)
    tp2 = peak
    if tp2 <= tp1:
        tp2 = max(peak, tp1 * 1.05)

    def box(level):
        return {"price": round(level, 4), "gain_pct": round((level - price) / price * 100, 1)}

    return {
        "entry_price": round(price, 4),
        "entry": round(price, 4),
        # منطقة دخول مسموحة: من القاع الفعلي حتى السعر الحالي (ما دام داخل 20% من القاع)
        "entry_low": round(trough_low, 4),
        "entry_high": round(price, 4),
        "stop_loss": round(stop, 4),
        "stop": round(stop, 4),
        "stop_basis": "القاع (Base Support) −2%" if stop >= trough_low * (1 - STOP_BUFFER) - 1e-9
                      else "أدنى قاع تشكّل بعد القمة −2%",
        "stop_pct": round((stop - price) / price * 100, 1),
        "target_1": round(tp1, 4),
        "target_2": round(tp2, 4),
        "tp1_basis": "آخر Lower High" if (lh and float(lh) > price * 1.01) else "فيبوناتشي 38.2%",
        "targets": [round(tp1, 4), round(tp2, 4)],
        "targets_detail": [box(tp1), box(tp2)],
        "final_100_target": round(peak, 4),
        "ready": bool(ready),
    }


# ---------------- درجة الجاهزية ----------------
def build_checklist(sig, ctx):
    """قائمة الفحص العربية (rule / points / status) — صيغة JSON موحدة."""
    pattern_rule = ctx.get("pattern_type") if sig.get("pattern") else "نمط قاع مزدوج أو قاع أعلى من قاع"
    if sig.get("volume") and ctx.get("volume_mode") == "uptick":
        volume_rule = "بداية زخم خفيف مع شمعة خضراء ضيقة فوق EMA9"
    else:
        volume_rule = "جفاف البيع وهدوء الفوليوم (RVOL <= 0.8x)"
    items = [
        ("near_base", "قريب من منطقة القاع (أقل من 15%)"),
        ("pattern", pattern_rule),
        ("stable", (f"ثبات الدعم لـ {ctx.get('stable_bars', 0)} جلسات" if sig.get("stable")
                    else "ثبات الدعم لجلستين متتاليتين أو أكثر")),
        ("rsi", "RSI يتحسن من التشبع البيعي"),
        ("volume", volume_rule),
        ("ema20", "استعادة EMA20 (الإغلاق فوق المتوسط)"),
        ("vwap", "السعر فوق VWAP"),
        ("macd", "تحسن MACD Histogram"),
        ("catalyst", "محفز مستقبلي قادم"),
    ]
    dist = float(ctx.get("distance_pct", 0) or 0)
    note = f"البعد الحالي عن القاع {dist:.1f}%"
    if not sig.get("near_base"):
        note += (" — تجاوز 20% فلا يُوصف السهم بأنه قريب من القاع" if dist > DIST_TOO_FAR * 100
                 else " — خارج نطاق 15% المطلوب للقرب من القاع")
    return [{"key": k, "rule": label, "points": POINTS[k], "status": bool(sig.get(k)),
             **({"note": note} if k == "near_base" else {})}
            for k, label in items]


def score_from_signals(sig):
    raw = sum(POINTS[k] for k in POINTS if sig.get(k))
    return int(raw)


def stage_from_score(score):
    if score >= STAGE_BANDS[STAGE_READY][0]:
        return STAGE_READY
    if score >= STAGE_BANDS[STAGE_SEMI][0]:
        return STAGE_SEMI
    return STAGE_WATCH


def float_label(v):
    if v is None:
        return "غير متوفر"
    try:
        v = float(v)
    except Exception:
        return "غير متوفر"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.0f}K"
    return f"{int(v)}"


def _ar_date(date_str, now=None):
    try:
        dt = pd.Timestamp(str(date_str)[:10])
    except Exception:
        return str(date_str)
    label = f"{dt.day} {AR_MONTHS.get(dt.month, '')}"
    today = pd.Timestamp(now.date()) if now is not None else pd.Timestamp.utcnow().normalize().tz_localize(None)
    if dt.year != today.year:
        label += f" {dt.year}"
    return label.strip()


# ---------------- التقييم الكامل ----------------
def evaluate_event(d, event, float_shares=None, news=None, intraday=None, now=None):
    """
    يقيّم حدث صعود مقابل شروط الاستراتيجية. يعيد None إذا استُبعد السهم.
    d        : شموع يومية
    event    : ناتج find_surge_event
    float_shares: عدد أسهم التداول الحر (None => استبعاد حسب الشرط الصارم)
    news     : ناتج news.fetch_news / news.analyze_news
    intraday : شموع 1h أو 4h لاكتشاف نمط القاع
    """
    if d is None or event is None or len(d) < RALLY_WINDOW + 2:
        return None

    n = len(d)
    price = float(d["Close"].iloc[-1])
    days_since_peak = n - 1 - int(event["event_idx"])
    base = float(event["base"])
    peak = float(event["high"])

    # ---- 4) فلتر انتهاء الصلاحية الزمنية ----
    if days_since_peak > MAX_DAYS_SINCE_PEAK:
        return None
    if int(event["event_idx"]) >= n - 1:
        return None                     # القمة نفسها في آخر جلسة: لا يوجد قاع بعدها ليقيسه الرادار
    # ---- 1) فلتر السعر ----
    if not MIN_PRICE <= price <= MAX_PRICE:
        return None
    # ---- 3) فلتر Float (يُستبعد إذا كان أكبر من 10M أو غير متوفر) ----
    if float_shares is None:
        return None
    if float(float_shares) > MAX_FLOAT:
        return None

    state = _post_peak_state(d, event)
    if state["real_breakdown"]:
        return None                                    # انهيار حقيقي => إلغاء الـ setup
    # لا بد أن السهم عاد فعلًا إلى منطقة القاع (ولمسها) بعد القمة
    if state["trough_low"] > base * (1 + NEAR_BASE):
        return None

    dist_pct = (price - base) / base * 100 if base > 0 else 0.0
    dist_ratio = dist_pct / 100.0

    # ---- المؤشرات ----
    tech = technicals(d)
    rsi_s = rsi(d["Close"].astype(float))
    rsi_now = float(rsi_s.iloc[-1])
    rsi_prev = float(rsi_s.iloc[-2])
    rsi_min_since_peak = float(rsi_s.iloc[int(event["event_idx"]):].min())
    close = d["Close"].astype(float)
    open_ = d["Open"].astype(float)
    highs = d["High"].astype(float).values
    lows = d["Low"].astype(float).values
    vol = d["Volume"].astype(float)
    rv = rvol_series(vol)
    rvol = float(rv.iloc[-1]) if pd.notna(rv.iloc[-1]) else 0.0

    # ---- 1) القرب من القاع (20 نقطة، وصفر نقاط إذا تجاوز 20%) ----
    sig_near_base = bool(dist_ratio <= NEAR_BASE)
    if dist_ratio > DIST_TOO_FAR:
        sig_near_base = False

    # ---- 2) نمط القاع على فريم 1h/4h (15 نقطة) ----
    pattern = detect_base_pattern(intraday, base) or detect_base_pattern(d.tail(PATTERN_LOOKBACK), base)
    sig_pattern = pattern is not None
    pattern_type = pattern["type"] if pattern else PATTERN_NONE

    # ---- 3) الثبات فوق الدعم (15 نقطة) ----
    sig_stable = bool(state["stable"])
    stable_bars = int(state["bars_since_trough"] + 1) if state["holding"] else 0

    # ---- 4) RSI يتحسن من التشبع البيعي (10 نقاط) ----
    # التشبع يُقرأ من أدنى RSI بعد القمة على اليومي، أو من أدنى RSI على فريم 1h/4h
    # أثناء الجفاف/التصفية (Wilder RSI(14) على اليومي بطيء وقد لا يهبط تحت 30 خلال 20 جلسة).
    rsi_min_intraday = None
    if intraday is not None and len(intraday) > 20:
        rsi_i = rsi(intraday["Close"].astype(float))
        seg_i = rsi_i.dropna().tail(40)
        if len(seg_i):
            rsi_min_intraday = float(seg_i.min())
    was_oversold = bool(rsi_min_since_peak < 30 or (rsi_min_intraday is not None and rsi_min_intraday < 30))
    sig_rsi = bool(was_oversold and rsi_now > 35 and rsi_now > rsi_prev)

    # ---- 5) جفاف الفوليوم أو بداية زخم خفيف (10 نقاط) ----
    body = abs(float(close.iloc[-1]) - float(open_.iloc[-1])) / float(open_.iloc[-1]) if open_.iloc[-1] else 1.0
    rng = (float(highs[-1]) - float(lows[-1])) / price if price else 1.0
    green_tight = bool(close.iloc[-1] >= open_.iloc[-1] and body <= TIGHT_BODY_MAX and rng <= TIGHT_RANGE_MAX)
    above_ema9 = bool(close.iloc[-1] > tech["ema9"])
    dry = bool(rvol <= RVOL_DRY) and state["holding"]
    uptick = bool(RVOL_UPTICK[0] <= rvol <= RVOL_UPTICK[1] and green_tight and above_ema9)
    sig_volume = bool(dry or uptick)
    volume_mode = "dry" if dry else ("uptick" if uptick else "none")

    # ---- 6) EMA20 (10 نقاط) ----
    sig_ema20 = bool(tech["ema20_reclaim"])

    # ---- 7) VWAP (5 نقاط) — VWAP مرتكز من قاع الثبات ----
    vwap = anchored_vwap(d, state["trough_abs"])
    sig_vwap = bool(price > vwap)

    # ---- 8) MACD (5 نقاط) ----
    sig_macd = bool(tech["macd_hist"] > 0 or tech["macd_improving"])

    # ---- 9) محفز مستقبلي (10 نقاط) ----
    news_flags = summarize_news(news, now=now,
                                min_days=CATALYST_MIN_DAYS, max_days=CATALYST_MAX_DAYS)
    sig_catalyst = bool(news_flags["has_upcoming_catalyst"])

    sig = {
        "near_base": sig_near_base, "pattern": sig_pattern, "stable": sig_stable,
        "rsi": sig_rsi, "volume": sig_volume, "ema20": sig_ema20,
        "vwap": sig_vwap, "macd": sig_macd, "catalyst": sig_catalyst,
    }
    score = score_from_signals(sig)
    stage = stage_from_score(score)
    ready = stage == STAGE_READY

    ctx = {"pattern_type": pattern_type, "stable_bars": stable_bars, "volume_mode": volume_mode,
           "distance_pct": dist_pct}
    checklist = build_checklist(sig, ctx)
    missing = [MISSING_LABELS[k] for k in POINTS if not sig.get(k)]

    tech.update({
        "vwap": round(vwap, 4),
        "vwap_reclaim": sig_vwap,
        "rvol": round(rvol, 2),
        "volume_ok": sig_volume,
        "volume_mode": volume_mode,
        "rsi_recovery_signal": sig_rsi,
        "rsi_min_since_peak": round(rsi_min_since_peak, 2),
        "rsi_min_intraday": round(rsi_min_intraday, 2) if rsi_min_intraday is not None else None,
        "higher_low": bool(pattern and pattern["type"] == PATTERN_HIGHER_LOW),
        "no_new_low": bool(state["bars_since_trough"] >= 2),
        "stable_bars": stable_bars,
        "lower_high_level": round(state["lower_high"], 4) if state.get("lower_high") else None,
    })

    plan = _entry_plan(d, event, state, ready)

    return {
        # ---- صيغة JSON المطلوبة ----
        "price": round(price, 4),
        "float": float_label(float_shares),
        "stage": stage,
        "readiness_score": int(score),
        "days_since_peak": int(days_since_peak),
        "base_support": round(base, 4),
        "distance_from_base_pct": round(dist_pct, 2),
        "has_upcoming_catalyst": sig_catalyst,
        "catalyst_detail": news_flags["catalyst_detail"],
        "has_warning": bool(news_flags["has_warning"]),
        "warning_detail": news_flags["warning_detail"],
        "pattern_type": pattern_type,
        "indicators": {
            "rsi": round(rsi_now, 2), "rsi_prev": round(rsi_prev, 2),
            "ema9": tech["ema9"], "ema20": tech["ema20"], "vwap": round(vwap, 4),
            "rvol": round(rvol, 2), "macd_hist": tech["macd_hist"],
        },
        "checklist": checklist,
        "missing_conditions": missing,
        # ---- حقول مساندة (بنفس المفاتيح الإنجليزية) ----
        "event_date": event["event_date"],
        "event_price": round(float(event["event_price"]), 4),
        "prior_base": round(base, 4),
        "prior_high": round(peak, 4),
        "prior_rally_pct": round(float(event["rally_pct"]), 2),
        "surge_pct": round(float(event["rally_pct"]), 2),
        "max_drawdown_pct": round(state["max_drawdown_pct"], 2),
        "drawdown_pct": round((peak - price) / peak * 100, 2) if peak else 0.0,
        "float_shares": int(float_shares),
        "support": {
            "level": round(base, 4),
            "trough_low": round(float(state["trough_low"]), 4),
            "stable_sessions": int(state["bars_since_trough"]),
            "stable_bars": stable_bars,
            "near_support": sig_near_base,
            "distance_pct": round(dist_pct, 2),
            "tests": int(sum(1 for lw in lows[int(event["event_idx"]) + 1:] if lw <= base * (1 + ZONE_TOL))),
            "false_breakdown": bool(state["false_breakdown"]),
            "holding": bool(state["holding"]),
        },
        "technical": tech,
        "signal_flags": sig,
        "points": {k: (POINTS[k] if sig.get(k) else 0) for k in POINTS},
        "pattern": pattern,
        "plan": plan,
        "status": stage,                     # توافق مع الواجهة الحالية
        "ready": ready,
        "semi_ready": stage == STAGE_SEMI,
        "news": news or {"warnings": [], "catalysts": [], "temp_excluded": False},
    }


def analyze(d, float_shares=None, news=None, intraday=None, now=None):
    event = find_surge_event(d)
    return evaluate_event(d, event, float_shares, news, intraday, now) if event else None


def chart_data(d, n=ANALYSIS_DAYS):
    x = d.tail(n).copy()
    rrsi = rsi(x["Close"].astype(float))
    return [{
        "date": str(idx.date()),
        "open": round(float(row.Open), 4),
        "high": round(float(row.High), 4),
        "low": round(float(row.Low), 4),
        "close": round(float(row.Close), 4),
        "volume": int(row.Volume) if pd.notna(row.Volume) else 0,
        "rsi": round(float(rrsi.loc[idx]), 2) if pd.notna(rrsi.loc[idx]) else None,
    } for idx, row in x.iterrows()]
