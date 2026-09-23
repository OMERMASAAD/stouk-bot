# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

MIN_PRICE, MAX_PRICE = 1.0, 5.0
MAX_WATCH_DAYS = 20
MIN_PRIOR_RALLY_PCT = 100.0
MIN_DRAWDOWN_PCT = 50.0   # حد أدنى تقريبي للهبوط؛ والشرط الفعلي هو العودة لمنطقة الدعم قبل الصعود
SUPPORT_TOL = 0.08
BREAK_TOL = 0.03
MIN_SUPPORT_SESSIONS = 3
ANALYSIS_DAYS = 30

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1/period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def macd(series):
    fast, slow = ema(series, 12), ema(series, 26)
    line = fast - slow
    signal = ema(line, 9)
    return line, signal, line - signal

def find_surge_event(d):
    # Day 0 is the peak of a >100% rally. Only peaks from the last 20 sessions qualify.
    if len(d) < 20:
        return None
    end = len(d) - 1
    first_peak = max(19, end - MAX_WATCH_DAYS)
    candidates = []
    for peak_idx in range(first_peak, end + 1):
        close_at_peak = float(d["Close"].iloc[peak_idx])
        if not MIN_PRICE <= close_at_peak <= MAX_PRICE:
            continue
        start = peak_idx - 19
        window = d.iloc[start:peak_idx + 1]
        base_label = window["Low"].astype(float).idxmin()
        base_idx = d.index.get_loc(base_label)
        base = float(d.loc[base_label, "Low"])
        high = float(d["High"].iloc[peak_idx])
        rally = (high - base) / base * 100 if base > 0 else 0
        if rally >= MIN_PRIOR_RALLY_PCT:
            candidates.append({
                "event_idx": peak_idx,
                "event_date": str(d.index[peak_idx].date()),
                "event_price": close_at_peak,
                "base": base,
                "base_idx": base_idx,
                "high": high,
                "high_idx": peak_idx,
                "rally_pct": rally
            })
    return max(candidates, key=lambda x: x["event_idx"]) if candidates else None

def support_stats(d, base):
    x = d.tail(ANALYSIS_DAYS)
    tests = stable = best_stable = 0
    for low, close in zip(x["Low"].astype(float), x["Close"].astype(float)):
        if low < base * (1 - BREAK_TOL):
            stable = 0
        elif base * (1 - SUPPORT_TOL) <= low <= base * (1 + SUPPORT_TOL) and close >= base * (1 - SUPPORT_TOL):
            tests += 1
            stable += 1
            best_stable = max(best_stable, stable)
        else:
            stable = 0
    return tests, best_stable

def technicals(d):
    x = d.tail(ANALYSIS_DAYS)
    close, volume = x["Close"].astype(float), x["Volume"].astype(float)
    rrsi = rsi(close)
    e20, e50 = ema(close, 20), ema(close, 50)
    ml, ms, mh = macd(close)
    avg_vol20 = volume.rolling(20).mean()
    vol_ratio = volume.iloc[-1] / avg_vol20.iloc[-1] if avg_vol20.iloc[-1] else 0
    r, r_prev = float(rrsi.iloc[-1]), float(rrsi.iloc[-2])
    mh_now, mh_prev = float(mh.iloc[-1]), float(mh.iloc[-2])
    rsi_recovery = bool(r > 30 and r > r_prev)
    macd_improving = bool(mh_now > mh_prev)
    positive = sum([
        rsi_recovery, macd_improving,
        bool(close.iloc[-1] > e20.iloc[-1] and e20.iloc[-1] > e20.iloc[-4]),
        bool(vol_ratio >= 1.2), bool(close.iloc[-1] > e50.iloc[-1])
    ])
    return {
        "rsi": round(r,2), "rsi_prev": round(r_prev,2),
        "rsi_oversold_recent": bool(rrsi.min() < 30), "rsi_recovery": rsi_recovery,
        "ema20": round(float(e20.iloc[-1]),4), "ema50": round(float(e50.iloc[-1]),4),
        "ema20_reclaim": bool(close.iloc[-1] > e20.iloc[-1]),
        "ema20_rising": bool(e20.iloc[-1] > e20.iloc[-4]),
        "ema50_reclaim": bool(close.iloc[-1] > e50.iloc[-1]),
        "macd": round(float(ml.iloc[-1]),5), "macd_signal": round(float(ms.iloc[-1]),5),
        "macd_hist": round(mh_now,5),
        "macd_cross": bool(ml.iloc[-1] > ms.iloc[-1] and ml.iloc[-2] <= ms.iloc[-2]),
        "macd_improving": macd_improving,
        "volume_ratio": round(float(vol_ratio),2), "volume_confirm": bool(vol_ratio >= 1.2),
        "positive_confirmations": positive
    }

def make_plan(d, base, prior_high):
    price = float(d["Close"].iloc[-1])
    resist = []
    for x in d["High"].astype(float).tail(ANALYSIS_DAYS):
        if x > price * 1.05 and all(abs(x-r)/r > .04 for r in resist):
            resist.append(float(x))
    targets = sorted(resist)[:3]
    if prior_high > price * 1.05 and all(abs(prior_high-r)/r > .04 for r in targets):
        targets.append(float(prior_high))
    return {
        "entry_low": round(base*.98,4),
        "entry_high": round(base*1.05,4),
        "stop": round(base*(1-BREAK_TOL),4),
        "targets": [round(x,4) for x in targets[:4]],
        "final_100_target": round(float(prior_high),4)
    }

def readiness_score(drawdown_pct, near_support, stable, tech, ready):
    # العرض فقط؛ لا يغيّر شروط الجاهزية.
    if ready:
        return 100
    score = 0
    if drawdown_pct >= 40:
        score += 20
    elif drawdown_pct >= 30:
        score += 15
    if near_support:
        score += 20
    if stable >= MIN_SUPPORT_SESSIONS:
        score += 20
    if tech["rsi_oversold_recent"]:
        score += 10
    if tech["rsi_recovery"]:
        score += 15
    if tech["macd_improving"]:
        score += 10
    if tech["positive_confirmations"] >= 3:
        score += 5
    return min(100, score)

def evaluate_event(d, event):
    age = len(d) - 1 - event["event_idx"]
    if age > MAX_WATCH_DAYS:
        return None

    price = float(d["Close"].iloc[-1])
    if not MIN_PRICE <= price <= MAX_PRICE:
        return None

    base = float(event["base"])
    prior_high = float(event["high"])
    drawdown_pct = ((prior_high - price) / prior_high * 100) if prior_high > 0 else 0

    # كسر الدعم بأكثر من 3% يلغي الحدث.
    if float(d["Low"].tail(ANALYSIS_DAYS).min()) < base * (1-BREAK_TOL):
        return None

    tests, stable = support_stats(d, base)
    near_support = base*(1-SUPPORT_TOL) <= price <= base*(1+SUPPORT_TOL)

    tech = technicals(d)
    recovery_core = tech["rsi_oversold_recent"] and tech["rsi_recovery"] and tech["macd_improving"]
    positive_stage = tech["positive_confirmations"] >= 3

    support_ready = near_support and stable >= MIN_SUPPORT_SESSIONS
    ready = (
        drawdown_pct >= MIN_DRAWDOWN_PCT
        and support_ready
        and recovery_core
        and positive_stage
    )
    semi = (
        drawdown_pct >= MIN_DRAWDOWN_PCT
        and (support_ready or near_support or tests >= 1)
        and not ready
    )

    if ready:
        status = "جاهز للدخول"
    elif semi:
        status = "شبه جاهز"
    else:
        status = "قيد المراقبة"

    missing = []
    if drawdown_pct < MIN_DRAWDOWN_PCT:
        missing.append("الهبوط من القمة أقل من 50%")
    if not near_support:
        missing.append("لم يعد السعر إلى منطقة الدعم")
    if tests < 1:
        missing.append("لم يختبر الدعم بعد")
    if stable < MIN_SUPPORT_SESSIONS:
        missing.append("ثبات الدعم أقل من 3 جلسات")
    if not tech["rsi_oversold_recent"]:
        missing.append("RSI لم يدخل التشبع البيعي تحت 30")
    if not tech["rsi_recovery"]:
        missing.append("RSI لم يبدأ التعافي")
    if not tech["macd_improving"]:
        missing.append("MACD Histogram لا يتحسن")
    if not positive_stage:
        missing.append("التأكيدات الفنية أقل من 3/5")

    score = readiness_score(drawdown_pct, near_support, stable, tech, ready)

    return {
        "event_date": event["event_date"],
        "event_price": round(event["event_price"],4),
        "surge_pct": round(event["rally_pct"],2),
        "watch_age": age,
        "price": round(price,4),
        "drawdown_pct": round(drawdown_pct,2),
        "prior_base": round(base,4),
        "prior_high": round(prior_high,4),
        "prior_rally_pct": round(event["rally_pct"],2),
        "support": {
            "tests": tests,
            "stable_sessions": stable,
            "near_support": near_support
        },
        "technical": tech,
        "readiness_score": score,
        "ready": ready,
        "semi_ready": semi,
        "status": status,
        "missing_conditions": missing,
        "plan": make_plan(d, base, prior_high)
    }

def analyze(d):
    event = find_surge_event(d)
    return evaluate_event(d, event) if event else None

def chart_data(d, n=30):
    x = d.tail(n).copy()
    rrsi = rsi(x["Close"].astype(float))
    return [{
        "date": str(idx.date()),
        "open": round(float(row.Open),4),
        "high": round(float(row.High),4),
        "low": round(float(row.Low),4),
        "close": round(float(row.Close),4),
        "volume": int(row.Volume) if pd.notna(row.Volume) else 0,
        "rsi": round(float(rrsi.loc[idx]),2) if pd.notna(rrsi.loc[idx]) else None
    } for idx,row in x.iterrows()]
