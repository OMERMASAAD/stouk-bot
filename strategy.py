# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

MIN_PRICE, MAX_PRICE = 1.0, 5.0
SURGE_PCT = 15.0
MIN_WATCH_DAYS, MAX_WATCH_DAYS = 2, 20
MIN_PRIOR_RALLY_PCT = 100.0
SUPPORT_TOL = 0.08
BREAK_TOL = 0.03
MIN_SUPPORT_SESSIONS = 3

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
    fast = ema(series, 12)
    slow = ema(series, 26)
    line = fast - slow
    signal = ema(line, 9)
    return line, signal, line - signal

def find_base_before(d, end_idx):
    start = max(0, end_idx - 70)
    window = d.iloc[start:end_idx]
    if len(window) < 20:
        return None
    lows = window["Low"].astype(float)
    base_label = lows.idxmin()
    base_idx = d.index.get_loc(base_label)
    base = float(d.loc[base_label, "Low"])
    high_slice = d.iloc[base_idx:end_idx+1]
    high = float(high_slice["High"].max())
    if base <= 0:
        return None
    rally = (high - base) / base * 100
    if rally < MIN_PRIOR_RALLY_PCT:
        return None
    return {"base": base, "base_idx": d.index.get_loc(base_idx), "high": high,
            "high_idx": end_idx, "rally_pct": rally}

def find_surge_events(d):
    events = []
    if len(d) < 60:
        return events
    closes = d["Close"].astype(float)
    changes = closes.pct_change() * 100
    start = max(1, len(d) - (MAX_WATCH_DAYS + 1))
    for i in range(start, len(d) - 1):
        price = float(closes.iloc[i])
        if not MIN_PRICE <= price <= MAX_PRICE or float(changes.iloc[i]) < SURGE_PCT:
            continue
        base = find_base_before(d, i)
        if base:
            events.append({
                "event_idx": i,
                "event_date": str(d.index[i].date()),
                "event_price": price,
                "event_pct": float(changes.iloc[i]),
                **base
            })
    return events

def support_stats(d, base):
    lows = d["Low"].astype(float).tail(MAX_WATCH_DAYS)
    closes = d["Close"].astype(float).tail(MAX_WATCH_DAYS)
    tests = 0
    best_stable = 0
    stable = 0
    for low, close in zip(lows, closes):
        if low < base * (1 - BREAK_TOL):
            stable = 0
            continue
        if base * (1 - SUPPORT_TOL) <= low <= base * (1 + SUPPORT_TOL) and close >= base * (1 - SUPPORT_TOL):
            tests += 1
            stable += 1
            best_stable = max(best_stable, stable)
        else:
            stable = 0
    return tests, best_stable

def technicals(d):
    close = d["Close"].astype(float)
    high = d["High"].astype(float)
    low = d["Low"].astype(float)
    volume = d["Volume"].astype(float)
    rrsi = rsi(close)
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    ml, ms, mh = macd(close)

    avg_vol20 = volume.rolling(20).mean()
    vol_ratio = volume.iloc[-1] / avg_vol20.iloc[-1] if avg_vol20.iloc[-1] else 0
    r = float(rrsi.iloc[-1])
    r_prev = float(rrsi.iloc[-2]) if len(rrsi) > 1 else r
    macd_hist = float(mh.iloc[-1])
    macd_hist_prev = float(mh.iloc[-2]) if len(mh) > 1 else macd_hist
    macd_cross = bool(ml.iloc[-1] > ms.iloc[-1] and ml.iloc[-2] <= ms.iloc[-2])
    ema20_reclaim = bool(close.iloc[-1] > e20.iloc[-1])
    ema20_rising = bool(e20.iloc[-1] > e20.iloc[-4]) if len(e20) >= 5 else False
    ema50_reclaim = bool(close.iloc[-1] > e50.iloc[-1])
    oversold_recent = bool(rrsi.tail(8).min() < 30)
    rsi_recovery = bool(r > 30 and r > r_prev)
    macd_improving = bool(macd_hist > macd_hist_prev)
    volume_confirm = bool(vol_ratio >= 1.2)
    positive = sum([rsi_recovery, macd_improving, ema20_reclaim and ema20_rising, volume_confirm, ema50_reclaim])

    return {
        "rsi": round(r, 2),
        "rsi_prev": round(r_prev, 2),
        "rsi_oversold_recent": oversold_recent,
        "rsi_recovery": rsi_recovery,
        "ema20": round(float(e20.iloc[-1]), 4),
        "ema50": round(float(e50.iloc[-1]), 4),
        "ema20_reclaim": ema20_reclaim,
        "ema20_rising": ema20_rising,
        "ema50_reclaim": ema50_reclaim,
        "macd": round(float(ml.iloc[-1]), 5),
        "macd_signal": round(float(ms.iloc[-1]), 5),
        "macd_hist": round(macd_hist, 5),
        "macd_cross": macd_cross,
        "macd_improving": macd_improving,
        "volume_ratio": round(float(vol_ratio), 2),
        "volume_confirm": volume_confirm,
        "positive_confirmations": positive
    }

def make_plan(d, base, prior_high):
    price = float(d["Close"].iloc[-1])
    entry_low = base * 0.98
    entry_high = base * 1.05
    stop = base * (1 - BREAK_TOL)
    resist = []
    for x in d["High"].astype(float).tail(120):
        if x > price * 1.05 and all(abs(x-r)/r > .04 for r in resist):
            resist.append(float(x))
    targets = sorted(resist)[:3]
    if prior_high > price * 1.05 and all(abs(prior_high-r)/r > .04 for r in targets):
        targets.append(float(prior_high))
    return {
        "entry_low": round(entry_low, 4),
        "entry_high": round(entry_high, 4),
        "stop": round(stop, 4),
        "targets": [round(x, 4) for x in targets[:4]],
        "final_100_target": round(float(prior_high), 4)
    }

def evaluate_event(d, event):
    i = event["event_idx"]
    age = len(d) - 1 - i
    if not MIN_WATCH_DAYS <= age <= MAX_WATCH_DAYS:
        return None
    base = float(event["base"])
    price = float(d["Close"].iloc[-1])
    if not MIN_PRICE <= price <= MAX_PRICE:
        return None
    if float(d["Low"].tail(MAX_WATCH_DAYS).min()) < base * (1 - BREAK_TOL):
        return None

    tests, stable = support_stats(d, base)
    near_support = base * (1 - SUPPORT_TOL) <= price <= base * (1 + SUPPORT_TOL)
    tech = technicals(d)
    support_ready = near_support and stable >= MIN_SUPPORT_SESSIONS
    recovery_core = tech["rsi_oversold_recent"] and tech["rsi_recovery"] and tech["macd_improving"]
    positive_stage = tech["positive_confirmations"] >= 3
    ready = support_ready and recovery_core and positive_stage
    semi = support_ready or (near_support and tech["rsi_oversold_recent"] and tech["rsi_recovery"])

    if ready:
        status = "جاهز للدخول"
    elif semi:
        status = "شبه جاهز"
    else:
        status = "قيد المراقبة"

    return {
        "event_date": event["event_date"],
        "event_price": round(event["event_price"], 4),
        "surge_pct": round(event["event_pct"], 2),
        "watch_age": age,
        "price": round(price, 4),
        "prior_base": round(base, 4),
        "prior_high": round(event["high"], 4),
        "prior_rally_pct": round(event["rally_pct"], 2),
        "support": {"tests": tests, "stable_sessions": stable, "near_support": near_support},
        "technical": tech,
        "ready": ready,
        "semi_ready": semi,
        "status": status,
        "plan": make_plan(d, base, event["high"])
    }

def analyze(d):
    events = find_surge_events(d)
    if not events:
        return None
    results = [evaluate_event(d, e) for e in events]
    results = [x for x in results if x]
    if not results:
        return None
    priority = {"جاهز للدخول": 3, "شبه جاهز": 2, "قيد المراقبة": 1}
    return max(results, key=lambda x: (priority[x["status"]], x["technical"]["positive_confirmations"], x["watch_age"]))

def chart_data(d, n=45):
    x = d.tail(n).copy()
    rrsi = rsi(d["Close"].astype(float))
    out = []
    for idx, row in x.iterrows():
        out.append({
            "date": str(idx.date()),
            "open": round(float(row.Open), 4),
            "high": round(float(row.High), 4),
            "low": round(float(row.Low), 4),
            "close": round(float(row.Close), 4),
            "volume": int(row.Volume) if pd.notna(row.Volume) else 0,
            "rsi": round(float(rrsi.loc[idx]), 2) if pd.notna(rrsi.loc[idx]) else None
        })
    return out
