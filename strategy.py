# -*- coding: utf-8 -*-
"""
استراتيجية القاع → التشبع البيعي → الثبات → التعافي
البيانات: Yahoo Finance فقط (شموع يومية).

المنطق:
  قيد المتابعة : سعر $1-$4 + Float<=10M + صعود سابق + هبوط + عودة للقاع + RSI<30
  شبه جاهز     : ظهور علامات تحسن (نقاط >= 45) دون اشتراط كلها
  جاهز فنيًا   : تأكيد التعافي (نقاط >= 75)
الدخول عند القاع بعد التعافي: إن ارتفع السعر أكثر من +20% فوق القاع يخرج السهم (فات الدخول).
"""
import numpy as np
import pandas as pd

# ---------------- الإعدادات ----------------
MIN_PRICE, MAX_PRICE = 1.0, 4.0        # السعر الحالي فقط
MAX_FLOAT = 10_000_000                 # Float <= 10M
MAX_WATCH_DAYS = 40                    # أقصى عمر للحدث (جلسات من قمة الصعود) — يمنع بقاء setup قديم
MIN_PRIOR_RALLY_PCT = 100.0            # الصعود السابق الواضح: من القاع إلى القمة
MIN_DRAWDOWN_PCT = 30.0                # هبوط واضح من القمة
ZONE_TOL = 0.10                        # منطقة القاع = القاع ± 10%
NEAR_BASE = 0.15                       # "عاد إلى القاع" = لمس منطقة حتى +15% فوق القاع
BREAK_HARD = 0.15                      # كسر قوي للدعم
MAX_ENTRY_EXT = 0.20                   # الدخول عند القاع: إن ارتفع السعر أكثر من +20% فوق القاع فقد فات الدخول
STOP_BUFFER = 0.03                     # الوقف أسفل القاع بـ 3%
ANALYSIS_DAYS = 30

STAGE_WATCH = "قيد المتابعة"
STAGE_SEMI = "شبه جاهز"
STAGE_READY = "جاهز فنيًا"
STAGE_ORDER = {STAGE_READY: 3, STAGE_SEMI: 2, STAGE_WATCH: 1}
STAGE_BANDS = {STAGE_WATCH: (20, 44), STAGE_SEMI: (45, 74), STAGE_READY: (75, 100)}

# نقاط الجاهزية (المجموع 100)
POINTS = {
    "near_base": 20, "rsi": 10, "stable": 15, "higher_low": 15, "volume": 10,
    "ema20": 10, "macd": 5, "vwap": 5, "lh_break": 10,
}
SIGNAL_LABELS = {
    "rsi": "RSI لم يبدأ التحسن بعد من التشبع البيعي",
    "stable": "السعر ما زال يسجل قيعانًا جديدة (لم يثبت)",
    "higher_low": "لم يتكوّن Higher Low بعد",
    "volume": "الفوليوم لا يدعم التعافي بعد",
    "ema20": "السعر لم يستعد EMA20 بعد",
    "macd": "MACD Histogram لم يبدأ التحسن",
    "vwap": "لم يُستعد VWAP (المرتكز من القاع) بعد",
    "lh_break": "لم يُكسر آخر Lower High بعد",
}


# ---------------- المؤشرات ----------------
def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # لا خسائر إطلاقًا: RSI = 100 (أو 50 إن لم يوجد أي حركة)
    flat = np.where(ag > 0, 100.0, 50.0)
    return out.where(al != 0, pd.Series(flat, index=out.index))


def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def macd(series):
    fast, slow = ema(series, 12), ema(series, 26)
    line = fast - slow
    signal = ema(line, 9)
    return line, signal, line - signal


def technicals(d, n=None):
    """مؤشرات فنية على الشموع الممررة (n=None: كلها)."""
    x = d if n is None else d.tail(n)
    close, volume = x["Close"].astype(float), x["Volume"].astype(float)
    rrsi = rsi(close)
    e20, e50 = ema(close, 20), ema(close, 50)
    ml, ms, mh = macd(close)
    avg_vol20 = volume.rolling(20).mean()
    last_avg = avg_vol20.iloc[-1]
    vol_ratio = float(volume.iloc[-1] / last_avg) if pd.notna(last_avg) and last_avg else 0.0

    r, r_prev = float(rrsi.iloc[-1]), float(rrsi.iloc[-2])
    mh_now, mh_prev = float(mh.iloc[-1]), float(mh.iloc[-2])
    rsi_stable = bool(r >= r_prev)
    rsi_recovery = bool(r > 30 and r > r_prev)
    macd_improving = bool(mh_now > mh_prev)
    ema20_reclaim = bool(close.iloc[-1] > e20.iloc[-1])
    ema20_rising = bool(e20.iloc[-1] > e20.iloc[-4]) if len(e20) >= 4 else False

    positive = sum([
        rsi_recovery, macd_improving,
        bool(ema20_reclaim and ema20_rising),
        bool(vol_ratio >= 1.2), bool(close.iloc[-1] > e50.iloc[-1]),
    ])
    return {
        "rsi": round(r, 2), "rsi_prev": round(r_prev, 2),
        "rsi_oversold_recent": bool(rrsi.min() < 30),
        "rsi_stable": rsi_stable, "rsi_recovery": rsi_recovery,
        "ema20": round(float(e20.iloc[-1]), 4), "ema50": round(float(e50.iloc[-1]), 4),
        "ema20_reclaim": ema20_reclaim, "ema20_rising": ema20_rising,
        "ema50_reclaim": bool(close.iloc[-1] > e50.iloc[-1]),
        "macd": round(float(ml.iloc[-1]), 5), "macd_signal": round(float(ms.iloc[-1]), 5),
        "macd_hist": round(mh_now, 5),
        "macd_cross": bool(ml.iloc[-1] > ms.iloc[-1] and ml.iloc[-2] <= ms.iloc[-2]),
        "macd_improving": macd_improving,
        "volume_ratio": round(vol_ratio, 2), "volume_confirm": bool(vol_ratio >= 1.2),
        "positive_confirmations": int(positive),
    }


# ---------------- اكتشاف حدث الصعود ----------------
def find_surge_event(d):
    """
    يوم 0 = قمة صعود لا يقل عن 100% من قاع سابق (خلال 20 جلسة قبلها).
    القمة يجب أن تكون خلال آخر MAX_WATCH_DAYS جلسة وأعلى قمة حولها.
    السعر التاريخي للقمة لا يهم (قد يكون $7 أو $10).
    """
    if len(d) < 40:
        return None
    end = len(d) - 1
    first_peak = max(19, end - MAX_WATCH_DAYS)
    lows = d["Low"].astype(float).values
    highs = d["High"].astype(float).values
    best = None
    for p in range(first_peak, end + 1):
        # يجب ألا تكون هناك قمة أعلى منها في الجلسات الـ30 السابقة (نتجنب اعتبار قمة قديمة)
        if highs[p] < highs[max(0, p - 30):p].max():
            continue
        s = p - 19
        b = s + int(np.argmin(lows[s:p + 1]))
        base = float(lows[b])
        if base <= 0:
            continue
        rally = (float(highs[p]) - base) / base * 100
        if rally >= MIN_PRIOR_RALLY_PCT and (best is None or highs[p] > best["high"]):
            best = {
                "event_idx": p,
                "event_date": str(d.index[p].date()),
                "event_price": float(d["Close"].iloc[p]),
                "base": base, "base_idx": b,
                "high": float(highs[p]), "high_idx": p,
                "rally_pct": rally,
            }
    return best


# ---------------- أدوات الهيكل السعري ----------------
def _pivot_highs(highs, start, end):
    """مؤشرات القمم المحلية بين start و end (شاملة)."""
    out = []
    for i in range(max(1, start), min(end, len(highs) - 2) + 1):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
            out.append(i)
    return out


def _last_lower_high(highs, seg_start, bottom_abs):
    """آخر Lower High قبل القاع (آخر قمة محلية بين القمة الكبرى والقاع)."""
    if bottom_abs <= seg_start:
        return None
    piv = _pivot_highs(highs, seg_start, bottom_abs - 1)
    if piv:
        return float(highs[piv[-1]])
    return float(highs[seg_start:bottom_abs].max())


def _anchored_vwap(d, start_abs):
    x = d.iloc[start_abs:]
    tp = (x["High"].astype(float) + x["Low"].astype(float) + x["Close"].astype(float)) / 3
    vol = x["Volume"].astype(float)
    tot = vol.sum()
    return float((tp * vol).sum() / tot) if tot > 0 else float(x["Close"].iloc[-1])


def make_plan(d, base, prior_high, bottom_low=None):
    price = float(d["Close"].iloc[-1])
    highs = d["High"].astype(float).values
    n = len(highs)
    lo = max(0, n - 60)
    resist = []
    for i in _pivot_highs(highs, lo, n - 2):
        h = float(highs[i])
        if h > price * 1.05 and all(abs(h - r) / r > 0.04 for r in resist):
            resist.append(h)
    resist.sort()

    final = float(prior_high) if prior_high > price * 1.05 else None
    if final is not None:
        near = [r for r in resist if abs(r - final) / final > 0.04 and r < final][:2]
        targets = near + [final]
    else:
        targets = resist[:3]

    stop = base * (1 - STOP_BUFFER)
    if bottom_low is not None:
        stop = min(stop, float(bottom_low) * 0.98)
    stop = min(stop, price * 0.99)

    detail = [{"price": round(t, 4), "gain_pct": round((t - price) / price * 100, 1)} for t in targets]
    return {
        "entry": round(price, 4),
        # منطقة الدخول: من القاع الفعلي الذي تكوّن حتى السعر الحالي (قرب القاع بعد التعافي)
        "entry_low": round(float(bottom_low) if bottom_low is not None else base, 4),
        "entry_high": round(price, 4),
        "stop": round(stop, 4),
        "stop_pct": round((stop - price) / price * 100, 1),
        "targets": [t["price"] for t in detail],
        "targets_detail": detail,
        "final_100_target": round(float(prior_high), 4),
    }


# ---------------- الجاهزية ----------------
def readiness_points(sig):
    """sig: قاموس إشارات منطقية. يعيد (raw, points)."""
    pts = {k: (POINTS[k] if sig.get(k) else 0) for k in POINTS}
    return sum(pts.values()), pts


def stage_and_score(raw):
    if raw >= 75:
        stage = STAGE_READY
    elif raw >= 45:
        stage = STAGE_SEMI
    else:
        stage = STAGE_WATCH
    lo, hi = STAGE_BANDS[stage]
    return stage, int(min(hi, max(lo, raw)))


# ---------------- تقييم حدث ----------------
def evaluate_event(d, event, float_shares=None):
    """يعيد None إذا السهم خارج الرادار أو أُلغي الـ setup."""
    n = len(d)
    age = n - 1 - event["event_idx"]
    if age > MAX_WATCH_DAYS or age < 1:
        return None

    price = float(d["Close"].iloc[-1])
    if not MIN_PRICE <= price <= MAX_PRICE:
        return None
    if float_shares is not None and float_shares > MAX_FLOAT:
        return None

    base = float(event["base"])
    prior_high = float(event["high"])
    # الفكرة: الدخول عند القاع بعد التعافي من الهبوط، لا بعد أن يصعد السهم.
    # إذا ابتعد السعر أكثر من MAX_ENTRY_EXT فوق القاع فقد فات وقت الدخول ويخرج من الرادار.
    if price > base * (1 + MAX_ENTRY_EXT):
        return None
    drawdown_pct = (prior_high - price) / prior_high * 100 if prior_high > 0 else 0.0

    ev = int(event["event_idx"])
    close = d["Close"].astype(float)
    lows = d["Low"].astype(float).values
    since_lows = lows[ev + 1:]
    m = len(since_lows)
    bottom_rel = int(np.argmin(since_lows))
    bottom_abs = ev + 1 + bottom_rel
    bottom_low = float(since_lows[bottom_rel])
    sessions_since_bottom = m - 1 - bottom_rel

    # الهبوط المطلوب يُقاس حتى القاع (أقصى هبوط منذ القمة) لا حتى السعر الحالي،
    # لأن السهم قد يكون بدأ يتعافى بعد ذلك.
    max_drawdown_pct = (prior_high - bottom_low) / prior_high * 100 if prior_high > 0 else 0.0
    if max_drawdown_pct < MIN_DRAWDOWN_PCT:
        return None

    # ---- كسر الدعم: بسيط (False Breakdown) أم انهيار حقيقي ----
    zone_low = base * (1 - ZONE_TOL)
    below_zone = price < zone_low
    deep_break = bottom_low < base * (1 - BREAK_HARD)
    making_new_lows = sessions_since_bottom <= 1
    real_breakdown = below_zone and (deep_break or making_new_lows)
    if real_breakdown:
        return None
    false_breakdown = bottom_low < zone_low and not below_zone

    # ---- شرط العودة إلى القاع (لمس المنطقة منذ القمة) ----
    touched_base = bottom_low <= base * (1 + NEAR_BASE)
    # ---- التشبع البيعي: RSI < 30 منذ القمة ----
    rsi_s = rsi(close)
    rsi_min_since = float(rsi_s.iloc[ev + 1:].min())
    oversold = rsi_min_since < 30
    if not (touched_base and oversold):
        return None

    # ---- علامات التحسن ----
    tech = technicals(d.tail(120))
    tech["rsi_oversold_recent"] = True
    rsi_now = float(rsi_s.iloc[-1])
    rsi_improving = (rsi_now - rsi_min_since >= 3) and (rsi_now > float(rsi_s.iloc[-4]))
    no_new_low = sessions_since_bottom >= 2
    higher_low = (
        sessions_since_bottom >= 3
        and float(since_lows[-3:].min()) > bottom_low * 1.005
    )

    vol = d["Volume"].astype(float)
    avg20 = vol.rolling(20).mean().iloc[-1]
    rvol = float(vol.iloc[-1] / avg20) if pd.notna(avg20) and avg20 else 0.0
    prev10 = float(vol.iloc[-13:-3].mean()) if n >= 13 else 0.0
    selling_fades = prev10 > 0 and float(vol.iloc[-3:].mean()) < 0.8 * prev10
    volume_ok = bool((rvol >= 1.2 and close.iloc[-1] > close.iloc[-2]) or (selling_fades and no_new_low))

    ema20_reclaim = bool(tech["ema20_reclaim"])
    vwap = _anchored_vwap(d, bottom_abs)
    vwap_ok = bool(price > vwap or (price >= vwap * 0.97 and close.iloc[-1] > close.iloc[-2]))

    lh_level = _last_lower_high(d["High"].astype(float).values, ev + 1, bottom_abs)
    lh_break = bool(lh_level is not None and bottom_abs < n - 1 and price > lh_level)

    sig = {
        "near_base": True,
        "rsi": bool(rsi_improving),
        "stable": bool(no_new_low),
        "higher_low": bool(higher_low),
        "volume": volume_ok,
        "ema20": ema20_reclaim,
        "macd": bool(tech["macd_improving"]),
        "vwap": vwap_ok,
        "lh_break": lh_break,
    }
    raw, pts = readiness_points(sig)
    status, score = stage_and_score(raw)
    ready = status == STAGE_READY
    semi = status == STAGE_SEMI

    tech.update({
        "rsi_recovery_signal": bool(rsi_improving),
        "vwap": round(vwap, 4), "vwap_reclaim": vwap_ok,
        "higher_low": bool(higher_low), "no_new_low": bool(no_new_low),
        "lower_high_level": round(lh_level, 4) if lh_level else None,
        "lower_high_break": lh_break,
        "volume_ok": volume_ok, "rvol": round(rvol, 2),
    })

    missing = [SIGNAL_LABELS[k] for k in SIGNAL_LABELS if not sig.get(k)]
    dist_pct = (price - base) / base * 100 if base > 0 else 0.0

    return {
        "event_date": event["event_date"],
        "event_price": round(event["event_price"], 4),
        "surge_pct": round(event["rally_pct"], 2),
        "watch_age": age,
        "price": round(price, 4),
        "drawdown_pct": round(drawdown_pct, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
        "prior_base": round(base, 4),
        "prior_high": round(prior_high, 4),
        "prior_rally_pct": round(event["rally_pct"], 2),
        "support": {
            "tests": int(sum(1 for lw in since_lows if lw <= base * (1 + ZONE_TOL))),
            "stable_sessions": int(sessions_since_bottom),
            "near_support": bool(price <= base * (1 + NEAR_BASE)),
            "distance_pct": round(dist_pct, 2),
            "false_breakdown": bool(false_breakdown),
        },
        "technical": tech,
        "signal_flags": sig,
        "points": pts,
        "stage_flags": {"watch_stage": True, "semi_ready": semi or ready, "ready": ready},
        "readiness_raw": int(raw),
        "readiness_score": score,
        "ready": ready,
        "semi_ready": semi,
        "status": status,
        "missing_conditions": missing,
        "float_shares": float_shares,
        "plan": make_plan(d, base, prior_high, bottom_low),
    }


def analyze(d, float_shares=None):
    event = find_surge_event(d)
    return evaluate_event(d, event, float_shares) if event else None


def chart_data(d, n=30):
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
