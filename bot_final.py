# -*- coding: utf-8 -*-
"""
PULLBACK RECOVERY RADAR - BASE BOTTOM VERSION
الدعم = قاعدة بداية الصعود الأصلية، وليس قاع التصحيح الحالي.
Telegram فقط عند اكتمال إشارة الدخول.
"""
import os, warnings, requests, numpy as np, pandas as pd, yfinance as yf
from datetime import datetime
warnings.filterwarnings('ignore')

TELEGRAM_TOKEN=os.getenv('TELEGRAM_TOKEN','')
TELEGRAM_CHAT_ID=os.getenv('TELEGRAM_CHAT_ID','')

MIN_PRICE,MAX_PRICE=1.0,5.0
MIN_RALLY=100.0
MIN_SESSIONS,MAX_SESSIONS=4,20
MIN_AGE,MAX_AGE=4,31
SHORT_MAX=20_000
MIN_FLOAT,MAX_FLOAT=5_000_000,500_000_000
SUPPORT_TOL=0.12
BREAK_TOL=0.03
RSI_LOW=30.0
RSI_IMPROVE=1.5
CHUNK=100


def n(x):
    try:
        x=float(x)
        return None if np.isnan(x) or np.isinf(x) else x
    except: return None

def p(x):
    x=n(x); return 'غير متوفر' if x is None else f'${x:.2f}'
def mil(x):
    x=n(x); return 'غير متوفر' if x is None else f'{x/1e6:.1f} مليون'
def numfmt(x):
    x=n(x); return 'غير متوفر' if x is None else f'{x:,.0f}'

def clean(df):
    if df is None or df.empty: return pd.DataFrame()
    d=df.copy()
    if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
    d.columns=[str(x) for x in d.columns]
    cols=['Open','High','Low','Close','Volume']
    return d.dropna(subset=cols) if all(x in d.columns for x in cols) else pd.DataFrame()

def rsi(close,period=14):
    delta=close.diff(); gain=delta.clip(lower=0); loss=-delta.clip(upper=0)
    ag=gain.ewm(alpha=1/period,adjust=False).mean(); al=loss.ewm(alpha=1/period,adjust=False).mean()
    return 100-(100/(1+ag/al.replace(0,np.nan)))

def macd(close):
    a=close.ewm(span=12,adjust=False).mean(); b=close.ewm(span=26,adjust=False).mean(); m=a-b; s=m.ewm(span=9,adjust=False).mean()
    return m,s,m-s

def four_h(one_h):
    d=clean(one_h)
    if d.empty:return d
    try:
        return d.resample('4h').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'}).dropna()
    except:return pd.DataFrame()

def tickers():
    out=[]
    for url,col in [('https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt','Symbol'),('https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt','ACT Symbol')]:
        try:
            d=pd.read_csv(url,sep='|'); d=d[d['Test Issue']=='N']; out+=d[col].dropna().astype(str).tolist()
        except Exception as e: print('قائمة الأسهم:',e)
    return sorted(set(x.strip().upper() for x in out if x.strip().isalpha() and len(x.strip())<=5))

def detect_rally(d):
    d=clean(d)
    if len(d)<70:return None
    today=datetime.now().date(); best=None
    # القمة يجب أن تكون حديثة، ونبحث عن قاعدة قبلها بـ4-20 جلسة
    for end in range(max(20,len(d)-55),len(d)):
        hi=n(d.iloc[end].High)
        if hi is None:continue
        hdate=pd.Timestamp(d.index[end]).date(); age=(today-hdate).days
        if not MIN_AGE<=age<=MAX_AGE:continue
        for sessions in range(MIN_SESSIONS,MAX_SESSIONS+1):
            start=end-sessions
            if start<6:continue
            # قاعدة الصعود: آخر 5 جلسات قبل الانطلاق + أول يومين منه
            bw=d.iloc[max(0,start-5):start+2]
            base=n(bw.Low.min())
            if base is None or base<=0:continue
            pct=(hi-base)/base*100
            if pct<MIN_RALLY:continue
            # نريد أن تكون القاعدة قريبة من بعضها، وليس wick شاذًا جدًا
            med=n(bw.Low.median())
            if med is None or abs(base-med)/base>0.20:continue
            cand={'base':base,'high':hi,'pct':pct,'sessions':sessions,'high_date':hdate,'start_date':pd.Timestamp(d.index[start]).date(),'start_pos':start,'end_pos':end}
            if best is None or (cand['high_date'],cand['pct'])>(best['high_date'],best['pct']):best=cand
    return best

def support_check(d,h,base,rally):
    # القاع الأصلي هو المرجع.
    # نسمح بالسهم أن يلمس القاع أو يصنع قاعًا جديدًا قريبًا منه.
    low = base * (1 - SUPPORT_TOL)
    high = base * (1 + SUPPORT_TOL)

    after_daily = d[d.index.date >= rally['high_date']]
    if after_daily.empty:
        return False, 0, False

    # كسر واضح للقاع الأصلي يلغي الفرصة.
    lowest = n(after_daily.Low.min())
    if lowest is not None and lowest < base * (1 - BREAK_TOL):
        return False, 0, False

    x = four_h(h)
    x = x[x.index.date >= rally['high_date']].tail(60)
    if x.empty:
        return False, 0, False

    tests = []

    for idx, row in x.iterrows():
        lo = n(row.Low)
        cl = n(row.Close)

        if lo is None or cl is None:
            continue

        # اختبار فعلي لمنطقة القاع الأصلي.
        if low <= lo <= high:
            tests.append((idx, lo, cl))

    if not tests:
        return False, 0, False

    # آخر اختبار يجب أن يظهر رفضًا للكسر:
    # إغلاق فوق القاع أو ارتداد واضح من القاع.
    rejection = False

    for idx, lo, cl in tests[-10:]:
        rebound_from_low = cl >= lo * 1.02
        held_original_base = cl >= base * 0.98

        if rebound_from_low and held_original_base:
            rejection = True

    # السعر الحالي يجب أن يكون قريبًا من القاع،
    # وليس بعيدًا عنه بعد صعود جديد.
    current = n(x.Close.iloc[-1])
    near = current is not None and low <= current <= high

    return near, len(tests), rejection

def tech(h):
    x=four_h(h)
    if len(x)<55:return None
    x['RSI']=rsi(x.Close); x['M'],x['S'],x['H']=macd(x.Close); x['MA20']=x.Close.rolling(20).mean(); x['MA50']=x.Close.rolling(50).mean()
    rr=x.RSI.tail(3).dropna(); latest=x.iloc[-1]
    if rr.empty:return None
    m=n(latest.M); s=n(latest.S); hist=n(latest.H); prevh=n(x.H.iloc[-2]); r=n(latest.RSI); prevr=n(x.RSI.iloc[-2]); v=n(latest.Volume); va=n(x.Volume.tail(20).mean())
    return {'rsi':r,'oversold':rr.min()<RSI_LOW,'rsi_up':r is not None and prevr is not None and r-prevr>=RSI_IMPROVE,'macd_pos':m is not None and s is not None and m>s,'macd_up':hist is not None and prevh is not None and hist>prevh,'cross':m is not None and s is not None and n(x.M.iloc[-2]) is not None and n(x.S.iloc[-2]) is not None and m>s and n(x.M.iloc[-2])<=n(x.S.iloc[-2]),'ma20':n(latest.MA20),'ma50':n(latest.MA50),'above20':n(latest.MA20) is not None and latest.Close>latest.MA20,'above50':n(latest.MA50) is not None and latest.Close>latest.MA50,'volratio':(v/va if v is not None and va else 0)}

def confirm15(ticker):
    try:
        d=clean(yf.download(ticker,period='10d',interval='15m',auto_adjust=False,progress=False,threads=False))
        if len(d)<40:return False
        d['RSI']=rsi(d.Close); d['M'],d['S'],_=macd(d.Close)
        a=d.iloc[-1]; b=d.iloc[-2]; r=n(a.RSI); rp=n(b.RSI); m=n(a.M); s=n(a.S); v=n(a.Volume); av=n(d.Volume.tail(20).mean())
        return sum([r is not None and rp is not None and r>rp,m is not None and s is not None and m>s,v is not None and av and v>=av*1.05])>=2
    except:return False

def float_short(t):
    try:
        info=yf.Ticker(t).info
        return n(info.get('floatShares')),n(info.get('sharesShort'))
    except:return None,None

def news(t):
    keys=['approval','approved','contract','partnership','agreement','acquisition','merger','launch','trial','phase 3','phase 2','fda','revenue','earnings','profit','guidance','upgrade','order','deal']
    out=[]
    try:
        for z in (yf.Ticker(t).news or [])[:10]:
            c=z.get('content',{}); title=c.get('title','') if isinstance(c,dict) else z.get('title','')
            if any(k in str(title).lower() for k in keys):out.append(str(title))
    except:pass
    return out[:3]

def targets(d,base,hi,current):
    levels=[]
    for z in d.High.tail(100).dropna():
        z=n(z)
        if z and z>current*1.05 and z<current*4 and not any(abs(z-a)/a<.04 for a in levels):levels.append(z)
    if hi>current*1.05 and not any(abs(hi-a)/a<.04 for a in levels):levels.append(hi)
    diff=hi-base
    for z in [hi+diff*.272,hi+diff*.618,hi+diff,hi+diff*1.618]:
        if z>current*1.10 and not any(abs(z-a)/a<.04 for a in levels):levels.append(z)
    return sorted(levels)[:5]

def alert(r):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:return False
    ts='\n'.join(f'🎯 الهدف {i}: {p(x)}' for i,x in enumerate(r['targets'],1))
    ns='\n'.join('• '+x for x in r['news']) if r['news'] else 'لا يوجد محفز إيجابي واضح في البيانات المتاحة.'
    msg=(f"🚨 إشارة دخول من القاع 🚨\n\n📌 السهم: {r['ticker']}\n💵 السعر الحالي: {p(r['price'])}\n\n📈 الحركة السابقة:\n• الصعود: +{r['pct']:.1f}%\n• مدة الصعود: {r['sessions']} جلسات\n• قاع بداية الصعود: {p(r['base'])}\n• قمة الصعود: {p(r['high'])}\n\n🟢 منطقة الدخول حول القاع الأصلي (مرنة):\n• من {p(r['entry_low'])} إلى {p(r['entry_high'])}\n• الدعم الأصلي: {p(r['base'])}\n• وقف الخسارة: {p(r['stop'])}\n\n{ts}\n\n📊 تأكيدات 4 ساعات:\n• RSI: {r['rsi']:.1f}\n• RSI تحت 30 مؤخرًا: نعم\n• RSI يتحسن: نعم\n• MACD إيجابي: {'نعم' if r['macd_pos'] else 'لا'}\n• MACD يتحسن: {'نعم' if r['macd_up'] else 'لا'}\n• تقاطع MACD: {'نعم' if r['cross'] else 'لا'}\n• فوق MA20: {'نعم' if r['above20'] else 'لا'}\n• فوق MA50: {'نعم' if r['above50'] else 'لا'}\n• حجم 4H: {r['volratio']:.2f}x\n\n🛡️ الدعم:\n• اختبارات الدعم: {r['tests']}\n• رفض كسر الدعم: نعم\n• تأكيد 15 دقيقة: نعم\n\n📌 عوامل إضافية:\n• الفلوت: {mil(r['float'])}\n• الشورت: {numfmt(r['short'])}\n• الشورت ≤20 ألف: نعم\n\n📰 محفزات إيجابية:\n{ns}\n\n📍 الدخول على دفعات حول الدعم الأصلي، وليس مطاردة السهم بعد ابتعاده عن القاع.\n\n⚠️ تنبيه آلي فني وليس ضمانًا للربح. راجع الخبر والشارت وإدارة المخاطر.")
    try:
        u=f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'; q=requests.post(u,data={'chat_id':TELEGRAM_CHAT_ID,'text':msg},timeout=20); print('Telegram:',q.text,flush=True); return q.ok
    except Exception as e:print('Telegram error:',e);return False

def analyze(t):
    try:
        d=clean(yf.download(t,period='180d',interval='1d',auto_adjust=False,progress=False,threads=False))
        if len(d)<70:return None
        current=n(d.Close.iloc[-1])
        if current is None or not MIN_PRICE<=current<=MAX_PRICE:return None
        r=detect_rally(d)
        if not r:return None
        # أهم شرط: السعر الآن قرب قاعدة الصعود الأصلية
        entry_low=r['base']*.97; entry_high=r['base']*1.12
        if not entry_low<=current<=entry_high:return None
        h=clean(yf.download(t,period='30d',interval='1h',auto_adjust=False,progress=False,threads=False))
        if len(h)<150:return None
        near,tests,rejection=support_check(d,h,r['base'],r)
        if not near or tests<1 or not rejection:return None
        tx=tech(h)
        if not tx or not tx['oversold'] or not tx['rsi_up']:return None
        fl,sh=float_short(t)
        if sh is None or sh>SHORT_MAX:return None
        if fl is not None and not MIN_FLOAT<=fl<=MAX_FLOAT:return None
        if not confirm15(t):return None
        score=2+2+2+(1 if tx['macd_up'] else 0)+(1 if tx['macd_pos'] else 0)+(1 if tx['cross'] else 0)+(1 if tx['above20'] or tx['above50'] else 0)+(1 if tx['volratio']>=1.05 else 0)
        if score<6:return None
        tg=targets(d,r['base'],r['high'],current)
        if not tg:return None
        return {'ticker':t,'price':current,'base':r['base'],'high':r['high'],'pct':r['pct'],'sessions':r['sessions'],'entry_low':entry_low,'entry_high':entry_high,'stop':r['base']*(1-BREAK_TOL),'rsi':tx['rsi'],'macd_pos':tx['macd_pos'],'macd_up':tx['macd_up'],'cross':tx['cross'],'above20':tx['above20'],'above50':tx['above50'],'volratio':tx['volratio'],'tests':tests,'float':fl,'short':sh,'targets':tg,'news':news(t)}
    except Exception as e:
        print(f'[{t}] {e}',flush=True);return None

def scan():
    print('🚀 فحص اصطياد القاع الأصلي...',flush=True)
    ts=tickers(); print(f'تم تحميل {len(ts)} سهم',flush=True)
    found=0
    for i in range(0,len(ts),CHUNK):
        for t in ts[i:i+CHUNK]:
            r=analyze(t)
            if r:
                found+=1; key=(t,round(r['base'],4),round(r['high'],4))
                print(f"🔥 {t}: السعر {p(r['price'])} | الدعم الأصلي {p(r['base'])} | +{r['pct']:.1f}%",flush=True)
                if key not in ALERTED: 
                    if alert(r): ALERTED.add(key)
    print(f'انتهى الفحص. مرشحات الدخول: {found}',flush=True)

ALERTED=set()
if __name__=='__main__': scan()
