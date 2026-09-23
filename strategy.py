# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

MIN_PRICE, MAX_PRICE = 1.0, 5.0
TODAY_SURGE_PCT = 15.0
MIN_WATCH_DAYS, MAX_WATCH_DAYS = 2, 20
MIN_PRIOR_RALLY_PCT = 100.0
SUPPORT_TOL = 0.08
BREAK_TOL = 0.03
MIN_SUPPORT_SESSIONS = 3

def rsi(series, period=14):
    delta=series.diff()
    gain=delta.clip(lower=0)
    loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/period,adjust=False).mean()
    al=loss.ewm(alpha=1/period,adjust=False).mean()
    rs=ag/al.replace(0,np.nan)
    return 100-(100/(1+rs))

def find_prior_rally(d):
    best=None
    for hi in range(20,len(d)-5):
        high=float(d.High.iloc[hi])
        for bi in range(max(0,hi-50),hi-4):
            base=float(d.Low.iloc[bi])
            if base<=0: continue
            rally=(high-base)/base*100
            if rally>=MIN_PRIOR_RALLY_PCT:
                item={"base":base,"high":high,"base_idx":bi,"high_idx":hi,"rally_pct":rally}
                if best is None or hi>best["high_idx"]: best=item
    return best

def analyze(d):
    if d is None or len(d)<60: return None
    price=float(d.Close.iloc[-1])
    if not MIN_PRICE<=price<=MAX_PRICE: return None
    today_pct=(price/float(d.Close.iloc[-2])-1)*100
    if today_pct<TODAY_SURGE_PCT: return None
    rally=find_prior_rally(d)
    if not rally: return None
    age=len(d)-1-rally["high_idx"]
    if not MIN_WATCH_DAYS<=age<=MAX_WATCH_DAYS: return None
    base=rally["base"]
    lows=d.tail(MAX_WATCH_DAYS)
    tests=0; stable=0; best_stable=0
    for _,row in lows.iterrows():
        low=float(row.Low); close=float(row.Close)
        if low<base*(1-BREAK_TOL): stable=0; continue
        if base*(1-SUPPORT_TOL)<=low<=base*(1+SUPPORT_TOL) and close>=base*(1-SUPPORT_TOL):
            tests+=1; stable+=1; best_stable=max(best_stable,stable)
        else: stable=0
    near=base*(1-SUPPORT_TOL)<=price<=base*(1+SUPPORT_TOL)
    rr=rsi(d.Close).tail(6).dropna()
    oversold=len(rr)>=3 and rr.min()<30
    improving=len(rr)>=2 and rr.iloc[-1]>rr.iloc[-2]
    ready=near and best_stable>=MIN_SUPPORT_SESSIONS and oversold and improving
    entry_low=base*.98; entry_high=base*1.05; stop=base*(1-BREAK_TOL)
    resist=[]
    for x in d.High.tail(120):
        x=float(x)
        if x>price*1.05 and all(abs(x-r)/r>.04 for r in resist): resist.append(x)
    targets=sorted(resist)[:4]
    targets.append(entry_high*2)
    return {"price":round(price,4),"surge_pct":round(today_pct,2),"prior_base":round(base,4),
      "prior_high":round(rally["high"],4),"prior_rally_pct":round(rally["rally_pct"],2),"watch_age":age,
      "rsi":round(float(rr.iloc[-1]),2) if len(rr) else 0,"rsi_oversold":oversold,"rsi_improving":improving,
      "support":{"tests":tests,"stable_sessions":best_stable,"near_support":near},
      "ready":ready,"plan":{"entry_low":round(entry_low,4),"entry_high":round(entry_high,4),
      "stop":round(stop,4),"targets":[round(x,4) for x in targets]}}
