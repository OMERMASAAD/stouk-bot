import json,time
from datetime import datetime,timezone
import pandas as pd
import yfinance as yf
from strategy import analyze

def universe():
    out=set()
    for url,col in [("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt","Symbol"),("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt","ACT Symbol")]:
        try:
            d=pd.read_csv(url,sep="|"); d=d[d["Test Issue"]=="N"]; out.update(str(x).strip().upper() for x in d[col].dropna())
        except Exception as e: print(e)
    return sorted(x for x in out if x.isalpha() and len(x)<=5)

def fetch(t):
    try:
        d=yf.download(t,period="180d",interval="1d",auto_adjust=False,progress=False,threads=False)
        if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
        return d.dropna(subset=["Open","High","Low","Close","Volume"])
    except Exception: return pd.DataFrame()

rows=[]
for i,t in enumerate(universe(),1):
    r=analyze(fetch(t))
    if r: r["ticker"]=t; rows.append(r)
    if i%100==0: print("scanned",i)
    time.sleep(.03)
rows.sort(key=lambda x:(not x["ready"],-x["surge_pct"]))
with open("data.json","w",encoding="utf-8") as f:
    json.dump({"updated_at":datetime.now(timezone.utc).isoformat(),"count":len(rows),"signals":rows},f,ensure_ascii=False,indent=2)
print("signals",len(rows))
