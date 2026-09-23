# -*- coding: utf-8 -*-
# تجربة 10 أيام: القاع المزدوج + الرأس والكتفين المقلوب.
import pandas as pd
import numpy as np

MIN_PRICE, MAX_PRICE = 1.0, 5.0
MAX_WATCH_DAYS = 10
ANALYSIS_DAYS = 30
LOCAL_RADIUS = 2
DOUBLE_BOTTOM_TOL = 0.10
DOUBLE_NECK_MIN = 0.08
SHOULDER_TOL = 0.12
HEAD_EDGE = 0.05
BREAKOUT_TOL = 0.01


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


def technicals(d):
    x = d.tail(ANALYSIS_DAYS)
    close, volume = x["Close"].astype(float), x["Volume"].astype(float)
    rrsi = rsi(close)
    e20, e50 = ema(close, 20), ema(close, 50)
    ml, ms, mh = macd(close)
    avg_vol20 = volume.rolling(20).mean()

    r = float(rrsi.iloc[-1])
    r_prev = float(rrsi.iloc[-2]) if len(rrsi) > 1 else r
    mh_now = float(mh.iloc[-1])
    mh_prev = float(mh.iloc[-2]) if len(mh) > 1 else mh_now
    vol_ratio = float(volume.iloc[-1] / avg_vol20.iloc[-1]) if pd.notna(avg_vol20.iloc[-1]) and avg_vol20.iloc[-1] else 0

    return {
        "rsi": round(r, 2),
        "rsi_prev": round(r_prev, 2),
        "rsi_oversold_recent": bool(rrsi.tail(15).min() < 30),
        "rsi_recovery": bool(r > 30 and r > r_prev),
        "ema20": round(float(e20.iloc[-1]), 4),
        "ema50": round(float(e50.iloc[-1]), 4),
        "ema20_reclaim": bool(close.iloc[-1] > e20.iloc[-1]),
        "ema20_rising": bool(e20.iloc[-1] > e20.iloc[-4]) if len(e20) >= 4 else False,
        "ema50_reclaim": bool(close.iloc[-1] > e50.iloc[-1]),
        "macd": round(float(ml.iloc[-1]), 5),
        "macd_signal": round(float(ms.iloc[-1]), 5),
        "macd_hist": round(mh_now, 5),
        "macd_cross": bool(ml.iloc[-1] > ms.iloc[-1] and ml.iloc[-2] <= ms.iloc[-2]) if len(ml) > 1 else False,
        "macd_improving": bool(mh_now > mh_prev),
        "volume_ratio": round(vol_ratio, 2),
        "volume_confirm": bool(vol_ratio >= 1.2),
        "positive_confirmations": sum([
            bool(r > 30 and r > r_prev),
            bool(mh_now > mh_prev),
            bool(close.iloc[-1] > e20.iloc[-1] and e20.iloc[-1] > e20.iloc[-4]) if len(e20) >= 4 else False,
            bool(vol_ratio >= 1.2),
            bool(close.iloc[-1] > e50.iloc[-1])
        ])
    }


def local_lows(d):
    lows = d["Low"].astype(float).reset_index(drop=True).to_numpy()
    out = []
    for i in range(LOCAL_RADIUS, len(lows) - LOCAL_RADIUS):
        left = lows[i-LOCAL_RADIUS:i]
        right = lows[i+1:i+LOCAL_RADIUS+1]
        if lows[i] <= left.min() and lows[i] <= right.min():
            out.append(i)
    return out


def _window_high(d, a, b):
    if b <= a + 1:
        return None
    return float(d["High"].iloc[a+1:b].astype(float).max())


def find_double_bottom(d):
    if len(d) < 20:
        return None

    lows = local_lows(d)
    end = len(d) - 1
    candidates = []

    for p in range(len(lows)):
        i = lows[p]
        for q in range(p + 1, len(lows)):
            j = lows[q]
            if j - i < 3 or j - i > 12:
                continue
            if j > end - 1:
                continue

            l1 = float(d["Low"].iloc[i])
            l2 = float(d["Low"].iloc[j])
            avg = (l1 + l2) / 2
            if avg <= 0 or abs(l1 - l2) / avg > DOUBLE_BOTTOM_TOL:
                continue

            neckline = _window_high(d, i, j)
            if neckline is None:
                continue
            if neckline < max(l1, l2) * (1 + DOUBLE_NECK_MIN):
                continue

            # يجب أن يكون هناك ارتداد فعلي بعد القاع الثاني حتى نقول إن القاع بدأ يتكون.
            post = d.iloc[j:end+1]
            if post.empty or float(post["High"].max()) < l2 * 1.03:
                continue

            candidates.append({
                "event_idx": j,
                "event_date": str(d.index[j].date()),
                "event_price": float(d["Close"].iloc[j]),
                "pattern_type": "قاع مزدوج",
                "pattern_low": min(l1, l2),
                "left_low": l1,
                "head_low": None,
                "right_low": l2,
                "neckline": neckline,
                "pattern_height": neckline - min(l1, l2),
                "pattern_low_idx": j,
                "pattern_high": neckline,
                "formation_start_idx": i,
                "formation_end_idx": j
            })

    return max(candidates, key=lambda x: x["event_idx"]) if candidates else None


def find_inverse_head_shoulders(d):
    if len(d) < 22:
        return None

    lows = local_lows(d)
    end = len(d) - 1
    candidates = []

    for a in range(len(lows)):
        ls = lows[a]
        for b in range(a + 1, len(lows)):
            head = lows[b]
            if head - ls < 2 or head - ls > 9:
                continue
            for cidx in range(b + 1, len(lows)):
                rs = lows[cidx]
                if rs - head < 2 or rs - head > 9 or rs > end - 1:
                    continue

                left = float(d["Low"].iloc[ls])
                h = float(d["Low"].iloc[head])
                right = float(d["Low"].iloc[rs])

                if h > min(left, right) * (1 - HEAD_EDGE):
                    continue

                shoulders_avg = (left + right) / 2
                if shoulders_avg <= 0 or abs(left-right) / shoulders_avg > SHOULDER_TOL:
                    continue

                n1 = _window_high(d, ls, head)
                n2 = _window_high(d, head, rs)
                if n1 is None or n2 is None:
                    continue

                neckline = (n1 + n2) / 2
                post = d.iloc[rs:end+1]
                if post.empty or float(post["High"].max()) < right * 1.03:
                    continue

                candidates.append({
                    "event_idx": rs,
                    "event_date": str(d.index[rs].date()),
                    "event_price": float(d["Close"].iloc[rs]),
                    "pattern_type": "رأس وكتفين مقلوب",
                    "pattern_low": h,
                    "left_low": left,
                    "head_low": h,
                    "right_low": right,
                    "neckline": neckline,
                    "pattern_height": neckline - h,
                    "pattern_low_idx": head,
                    "pattern_high": neckline,
                    "formation_start_idx": ls,
                    "formation_end_idx": rs
                })

    return max(candidates, key=lambda x: x["event_idx"]) if candidates else None


def find_pattern_event(d):
    events = [x for x in [find_double_bottom(d), find_inverse_head_shoulders(d)] if x]
    return max(events, key=lambda x: x["event_idx"]) if events else None


def pattern_valid_now(d, event):
    low = float(d["Low"].iloc[-1])
    if event["pattern_type"] == "قاع مزدوج":
        invalid = min(event["left_low"], event["right_low"]) * 0.97
    else:
        invalid = event["head_low"] * 0.97
    return low >= invalid


def readiness_score(pattern_formed, near_neckline, breakout, tech, h4=None):
    score = 35 if pattern_formed else 15
    if near_neckline:
        score += 15
    if breakout:
        score += 25
    if tech["rsi_oversold_recent"]:
        score += 5
    if tech["rsi_recovery"]:
        score += 8
    if tech["macd_improving"]:
        score += 7
    if tech["volume_confirm"]:
        score += 5
    if h4:
        score += 5 if h4.get("rsi_recovery") or h4.get("macd_improving") else 0
    return min(100, score)


def make_plan(event):
    neckline = float(event["neckline"])
    height = max(float(event["pattern_height"]), 0)
    target = neckline + height
    invalid = min(float(event["left_low"]), float(event["right_low"]), float(event["pattern_low"])) * 0.97

    return {
        "entry_low": round(neckline * 1.005, 4),
        "entry_high": round(neckline * 1.03, 4),
        "stop": round(invalid, 4),
        "targets": [
            round(neckline + height * 0.5, 4),
            round(target, 4)
        ],
        "final_100_target": round(target, 4)
    }


def evaluate_event(d, event, h4=None):
    age = len(d) - 1 - int(event["event_idx"])
    if age < 0 or age > MAX_WATCH_DAYS:
        return None

    price = float(d["Close"].iloc[-1])
    if not MIN_PRICE <= price <= MAX_PRICE:
        return None

    if not pattern_valid_now(d, event):
        return None

    neckline = float(event["neckline"])
    breakout = price >= neckline * (1 + BREAKOUT_TOL)
    near_neckline = price >= neckline * 0.95

    # القاع/الكتف الأيمن يجب أن يكون قد ارتد بالفعل، وإلا لا نعتبر النموذج متكونًا.
    formed = float(d["High"].iloc[int(event["event_idx"]):].max()) >= float(event["right_low"]) * 1.03
    if not formed:
        return None

    tech = technicals(d)
    h4 = h4 or {}
    ready = bool(
        breakout
        and tech["rsi_recovery"]
        and tech["macd_improving"]
        and (tech["volume_confirm"] or (h4.get("rsi_recovery") and h4.get("macd_improving")))
    )
    semi = bool(formed and (near_neckline or breakout) and not ready)

    status = "جاهز للدخول" if ready else ("شبه جاهز" if semi else "قيد المراقبة")
    score = readiness_score(formed, near_neckline, breakout, tech, h4)

    plan = make_plan(event)

    return {
        "event_date": event["event_date"],
        "event_price": round(event["event_price"], 4),
        "pattern_type": event["pattern_type"],
        "watch_age": age,
        "price": round(price, 4),
        "pattern_low": round(float(event["pattern_low"]), 4),
        "left_low": round(float(event["left_low"]), 4),
        "head_low": round(float(event["head_low"]), 4) if event.get("head_low") is not None else None,
        "right_low": round(float(event["right_low"]), 4),
        "neckline": round(neckline, 4),
        "pattern_height_pct": round((float(event["pattern_height"]) / float(event["pattern_low"]) * 100) if float(event["pattern_low"]) > 0 else 0, 2),
        "breakout": breakout,
        "pattern_formed": formed,
        "technical": tech,
        "technical_4h": h4,
        "readiness_score": score,
        "ready": ready,
        "semi_ready": semi,
        "status": status,
        "plan": plan
    }


def analyze(d, h4=None):
    event = find_pattern_event(d)
    return evaluate_event(d, event, h4) if event else None


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
        "rsi": round(float(rrsi.loc[idx]), 2) if pd.notna(rrsi.loc[idx]) else None
    } for idx, row in x.iterrows()]
