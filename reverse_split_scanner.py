# -*- coding: utf-8 -*-
"""رادار ما بعد التقسيم العكسي — التحليل اليومي أولاً."""
import argparse
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from news import fetch_news, summarize_news

DATA_FILE = "reverse_split_data.json"
MASTER_FILE = "master_low_float.json"
LOOKBACK_DAYS = 60
MAX_POST_SPLIT_RALLY = 20.0
MIN_DROP = 30.0
STABILITY_MIN = 3
STABILITY_MAX = 5


def download_daily(ticker):
    try:
        d = yf.download(ticker, period="5mo", interval="1d", auto_adjust=False, actions=True, progress=False, threads=False)
        if d is None or d.empty:
            return None
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        cols = ["Open", "High", "Low", "Close", "Volume"]
        d = d.dropna(subset=cols).copy()
        if "Stock Splits" not in d:
            d["Stock Splits"] = 0.0
        return d
    except Exception as exc:
        print("download error", ticker, exc)
        return None


def ema(s, n):
    return s.astype(float).ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.astype(float).diff()
    gain, loss = d.clip(lower=0), -d.clip(upper=0)
    ag, al = gain.ewm(alpha=1 / n, min_periods=n, adjust=False).mean(), loss.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    return 100 - 100 / (1 + ag / al.replace(0, np.nan))


def indicators(d):
    c, h, l, v = d.Close.astype(float), d.High.astype(float), d.Low.astype(float), d.Volume.astype(float)
    e12, e26 = ema(c, 12), ema(c, 26)
    macd = e12 - e26
    signal = ema(macd, 9)
    typical = (h + l + c) / 3
    vwap = (typical * v).rolling(20, min_periods=3).sum() / v.rolling(20, min_periods=3).sum().replace(0, np.nan)
    return {"ema20": float(ema(c, 20).iloc[-1]), "ema30": float(ema(c, 30).iloc[-1]), "ema50": float(ema(c, 50).iloc[-1]), "vwap": float(vwap.iloc[-1]), "rsi": float(rsi(c).iloc[-1]), "rsi_prev": float(rsi(c).iloc[-2]), "macd": float(macd.iloc[-1]), "macd_signal": float(signal.iloc[-1]), "macd_hist": float((macd - signal).iloc[-1]), "macd_hist_prev": float((macd - signal).iloc[-2]), "rvol": float(v.iloc[-1] / v.shift(1).rolling(20).mean().iloc[-1]) if v.shift(1).rolling(20).mean().iloc[-1] else 0.0}


def split_event(d, now):
    s = pd.to_numeric(d["Stock Splits"], errors="coerce").fillna(0)
    events = [(idx, float(v)) for idx, v in s.items() if float(v) > 1.0]
    if not events:
        return None
    idx, ratio = events[-1]
    dt = pd.Timestamp(idx).date()
    age = (now.date() - dt).days
    if age < 0 or age > LOOKBACK_DAYS:
        return None
    pos = d.index.get_loc(idx)
    if pos >= len(d) - 2:
        return None
    return {"date": str(dt), "ratio": ratio, "pos": int(pos), "age_days": int(age)}


def fit_wedge(d, start):
    x = d.iloc[max(start, len(d) - 32):].copy()
    if len(x) < 10:
        return None
    t = np.arange(len(x), dtype=float)
    hs, hi = np.polyfit(t, x.High.astype(float), 1)
    ls, li = np.polyfit(t, x.Low.astype(float), 1)
    upper_now, lower_now = hs * (len(x) - 1) + hi, ls * (len(x) - 1) + li
    gap_old = (hs * 2 + hi) - (ls * 2 + li)
    gap_now = upper_now - lower_now
    converging = hs < 0 and ls < 0 and gap_now > 0 and gap_now < max(gap_old, gap_now * 1.05)
    wedge = converging and ls < hs
    upper_break = float(d.Close.iloc[-1]) > upper_now * 1.005
    return {"detected": bool(wedge), "upper_slope": round(float(hs), 5), "lower_slope": round(float(ls), 5), "upper_now": round(float(upper_now), 4), "lower_now": round(float(lower_now), 4), "upper_break": bool(upper_break), "bars": int(len(x))}


def chart_rows(d):
    c = d.Close.astype(float)
    rr = rsi(c)
    e20, e30, e50 = ema(c, 20), ema(c, 30), ema(c, 50)
    m = ema(c, 12) - ema(c, 26); ms = ema(m, 9)
    typical = (d.High.astype(float) + d.Low.astype(float) + c) / 3
    vw = (typical * d.Volume.astype(float)).rolling(20, min_periods=3).sum() / d.Volume.astype(float).rolling(20, min_periods=3).sum().replace(0, np.nan)
    out=[]
    lows=d.Low.astype(float).tolist()
    for i,(idx,row) in enumerate(d.tail(90).iterrows()):
        j=d.index.get_loc(idx); out.append({"date":str(pd.Timestamp(idx).date()),"open":round(float(row.Open),4),"high":round(float(row.High),4),"low":round(float(row.Low),4),"close":round(float(row.Close),4),"volume":int(row.Volume),"ema20":round(float(e20.iloc[j]),4),"ema30":round(float(e30.iloc[j]),4),"ema50":round(float(e50.iloc[j]),4),"vwap":round(float(vw.iloc[j]),4) if pd.notna(vw.iloc[j]) else None,"rsi":round(float(rr.iloc[j]),2) if pd.notna(rr.iloc[j]) else None,"macd":round(float(m.iloc[j]),5),"macd_signal":round(float(ms.iloc[j]),5),"macd_hist":round(float((m-ms).iloc[j]),5),"swing_low":bool(0<i<len(lows)-1 and lows[j]<=lows[j-1] and lows[j]<=lows[min(j+1,len(lows)-1)])})
    return out


def evaluate(ticker, d, now=None, news=None):
    now = now or datetime.now(timezone.utc)
    ev = split_event(d, now)
    if not ev:
        return None
    p = ev["pos"]
    post = d.iloc[p:]
    split_open, split_high = float(d.Open.iloc[p]), float(d.High.iloc[p])
    peak = float(post.High.max())
    peak_pos = int(post.High.values.argmax())
    current = float(d.Close.iloc[-1])
    max_rally = (peak / split_open - 1) * 100 if split_open else 999
    drop = (peak - current) / peak * 100 if peak else 0
    if max_rally > MAX_POST_SPLIT_RALLY or drop < MIN_DROP:
        return None
    tech = indicators(d)
    recent_rsi = rsi(d.Close.astype(float)).iloc[p:]
    oversold = bool((recent_rsi < 30).any())
    rsi_recovery = bool(tech["rsi"] > tech["rsi_prev"] and tech["rsi"] < 50)
    under_ma = bool(current < tech["ema20"] and current < tech["ema30"] and current < tech["ema50"])
    under_vwap = bool(current < tech["vwap"])
    tail = d.iloc[-STABILITY_MAX:]
    base = float(tail.Low.min())
    support_span = (float(tail.High.max()) - base) / base if base else 99
    stable = bool(len(tail) >= STABILITY_MIN and support_span <= 0.12 and float(tail.Close.iloc[-1]) >= base * .97)
    stable_days = int(min(STABILITY_MAX, len(tail))) if stable else 0
    wedge = fit_wedge(d, p)
    news_summary = summarize_news(news or {"warnings": [], "catalysts": []}, now=now)
    if news_summary.get("has_warning"):
        return None
    score = (20 if drop >= 30 else 0) + (15 if oversold else 0) + (15 if stable else 0) + (10 if rsi_recovery else 0) + (10 if under_ma else 0) + (5 if under_vwap else 0) + (10 if wedge and wedge["detected"] else 0) + (10 if tech["macd_hist"] > tech["macd_hist_prev"] else 0) + (5 if max_rally <= 20 else 0)
    ready = bool(wedge and wedge["upper_break"] and oversold and rsi_recovery and stable)
    entry = wedge["upper_now"] if wedge else current
    stop = base * .98
    target1 = max(entry, wedge["upper_now"] + (wedge["upper_now"] - wedge["lower_now"]) if wedge else entry * 1.1)
    return {"ticker":ticker,"reverse_split":{"date":ev["date"],"ratio":ev["ratio"],"age_days":ev["age_days"],"opening_price":round(split_open,4),"split_day_high":round(split_high,4)},"price":round(current,4),"peak_after_split":round(peak,4),"peak_date":str(pd.Timestamp(post.index[peak_pos]).date()),"max_rally_pct":round(max_rally,2),"drop_pct":round(drop,2),"base_support":round(base,4),"stable_days":stable_days,"readiness_score":int(score),"stage":"جاهز فنيًا" if ready else ("شبه جاهز" if score>=55 else "قيد المتابعة"),"ready":ready,"conditions":{"drop_30":drop>=30,"rsi_oversold":oversold,"rsi_recovery":rsi_recovery,"below_ema20_30_50":under_ma,"below_vwap":under_vwap,"support_stable":stable,"macd_improving":tech["macd_hist"]>tech["macd_hist_prev"],"falling_wedge":bool(wedge and wedge["detected"]),"upper_break":bool(wedge and wedge["upper_break"])},"indicators":{k:round(v,5) for k,v in tech.items()},"wedge":wedge,"plan":{"entry":round(entry,4),"stop":round(stop,4),"target_1":round(target1,4),"target_main":round(split_high,4),"target_main_label":"قمة شمعة يوم التقسيم"},"chart":chart_rows(d),"news":news_summary}


def load_tickers(path=MASTER_FILE):
    try:
        with open(path,encoding="utf-8") as f: j=json.load(f)
        return [x["ticker"] for x in j.get("tickers",[]) if x.get("ticker")]
    except Exception:
        return []


def run(tickers=None, now=None):
    now=now or datetime.now(timezone.utc); tickers=tickers or load_tickers(); out=[]
    for n,t in enumerate(tickers,1):
        d=download_daily(t)
        if d is None: continue
        try:
            news=fetch_news(t, now=now)
            x=evaluate(t,d,now,news)
            if x: out.append(x)
        except Exception as exc: print("evaluate error",t,exc)
        if n%25==0: print("reverse split progress",n,"/",len(tickers))
    out.sort(key=lambda x:(x["ready"],x["readiness_score"],x["drop_pct"]),reverse=True)
    return {"updated_at":now.isoformat(),"count":len(out),"rules":{"lookback_days":LOOKBACK_DAYS,"max_post_split_rally_pct":MAX_POST_SPLIT_RALLY,"min_drop_pct":MIN_DROP,"stability_days":"3-5","timeframe":"daily","optional_confirmation":"4h when available"},"items":out}

if __name__ == "__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--tickers",default=""); ap.add_argument("--output",default=DATA_FILE); a=ap.parse_args()
    tickers=[x.strip().upper() for x in a.tickers.split(",") if x.strip()] or None
    with open(a.output,"w",encoding="utf-8") as f: json.dump(run(tickers),f,ensure_ascii=False,indent=2)
