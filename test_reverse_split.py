# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import reverse_split_scanner as rs


def build():
    n=95; idx=pd.bdate_range(end="2026-10-30",periods=n)
    close=[3.0]*50+[2.9,2.6,2.35,2.1,1.9,1.75,1.62,1.58,1.56,1.55,1.57,1.59,1.6,1.61,1.62,1.63,1.64,1.65,1.66,1.67,1.68,1.69,1.7,1.71,1.7,1.72,1.73,1.72,1.74,1.75,1.74,1.76,1.77,1.76,1.78,1.79,1.78,1.8,1.81,1.8,1.82,1.83,1.82,1.84,1.85]
    close=close[:n]; o=np.array(close)*1.01; h=np.maximum(o,close)*1.02; l=np.minimum(o,close)*.98; v=np.array([1000000]*n,dtype=float); v[50:]=500000
    d=pd.DataFrame({'Open':o,'High':h,'Low':l,'Close':close,'Volume':v,'Stock Splits':[0.0]*n},index=idx); d.iloc[50,d.columns.get_loc('Stock Splits')]=10.0; d.iloc[50,d.columns.get_loc('Open')]=3.0; d.iloc[50,d.columns.get_loc('High')]=3.4
    return d


def main():
    d=build(); now=pd.Timestamp('2026-10-30',tz='UTC').to_pydatetime()
    ev=rs.split_event(d,now); assert ev and ev['ratio']==10.0
    x=rs.evaluate('TEST',d,now,{'warnings':[],'catalysts':[]}); assert x and x['drop_pct']>=30 and x['max_rally_pct']<=20
    assert x['conditions']['rsi_oversold'] and x['conditions']['support_stable']
    assert isinstance(x['conditions']['below_ema20_30_50'], bool)
    assert x['plan']['target_main']==x['reverse_split']['split_day_high']
    old=d.copy(); old.index=old.index-pd.Timedelta(days=70); assert rs.split_event(old,now) is None
    print('✅ reverse split rules passed')

if __name__=='__main__': main()
