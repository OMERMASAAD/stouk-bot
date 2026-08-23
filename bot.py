# -*- coding: utf-8 -*-
"""
SIMPLE PULLBACK RECOVERY BOT
"""

import os
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import feedparser
import warnings

warnings.filterwarnings("ignore")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8808593618:AAEUz24M2638F7Al0ZHDJndmWIX4JCDLrJE")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1004436952886")

MIN_PRICE = 0.10
MAX_PRICE = 5.00
MIN_VOLUME = 100_000
MAX_SHORT_RATIO = 0.20
MAX_VOLATILITY = 0.10
RSI_THRESHOLD = 30
BATCH_SIZE = 50

def n(x):
    try:
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    except:
        return None

def p(x):
    x = n(x)
    if x is None:
        return "غير متوفر"
    return f"${x:.2f}"

def numfmt(x):
    x = n(x)
    if x is None:
        return "غير متوفر"
    return f"{x:,.0f}"

def calculate_rsi(closes, period=14):
    delta = pd.Series(closes).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi.iloc[-1] if len(rsi) > 0 else None

def calculate_ma(closes, period):
    return pd.Series(closes).rolling(window=period).mean().iloc[-1]

def get_historical_data(ticker, period="180d", interval="1d"):
    try:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
        return data
    except:
        return None

def check_last_5_support(data):
    if len(data) < 10:
        return False
    
    last_5 = data.tail(5)
    last_5_low = last_5['Low'].min()
    support = data.iloc[-6]['Low']
    
    if last_5_low < support * 0.97:
        return False
    
    return True

def check_4h_breakout(ticker):
    try:
        data_4h = yf.download(ticker, period="30d", interval="1h", auto_adjust=False, progress=False)
        
        if data_4h is None or len(data_4h) < 20:
            return False
        
        data_4h = data_4h.resample('4h').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}).dropna()
        
        if len(data_4h) < 3:
            return False
        
        last_3 = data_4h.tail(3)
        up_candles = sum(1 for i in range(len(last_3)) if last_3['Close'].iloc[i] > last_3['Open'].iloc[i])
        
        return up_candles >= 2
        
    except:
        return False

def check_volatility(data):
    if len(data) < 20:
        return False
    
    last_20 = data.tail(20)
    high_low_diff = (last_20['High'] - last_20['Low']) / last_20['Close']
    avg_volatility = high_low_diff.mean()
    
    return avg_volatility <= MAX_VOLATILITY

def check_news(ticker):
    try:
        url = f"https://feeds.bloomberg.com/markets/news.rss?ticker={ticker}"
        feed = feedparser.parse(url)
        
        if not feed.entries:
            return None
        
        latest = feed.entries[0]
        title = latest.title.lower()
        summary = latest.get('summary', '').lower()
        text = title + " " + summary
        
        positive_words = ["gains", "surge", "rally", "jump", "soars", "bullish", "approval", "deal", "partnership", "contract", "acquisition"]
        negative_words = ["falls", "drops", "decline", "loss", "crash", "bearish", "bankruptcy", "lawsuit", "recall"]
        
        pos_count = sum(1 for w in positive_words if w in text)
        neg_count = sum(1 for w in negative_words if w in text)
        
        if pos_count > neg_count:
            return "إيجابية"
        elif neg_count > pos_count:
            return "سلبية"
        else:
            return "محايدة"
            
    except:
        return None

def get_short_ratio(ticker):
    try:
        info = yf.Ticker(ticker).info
        short_ratio = info.get('shortPercentOfFloat', None)
        return short_ratio
    except:
        return None

def analyze_stock(ticker):
    try:
        print(f"🔍 فحص {ticker}...", flush=True)
        
        data = get_historical_data(ticker, period="180d", interval="1d")
        
        if data is None or len(data) < 50:
            return None
        
        current_price = n(data['Close'].iloc[-1])
        if current_price is None or not (MIN_PRICE <= current_price <= MAX_PRICE):
            return None
        
        volume = n(data['Volume'].iloc[-1])
        if volume is None or volume < MIN_VOLUME:
            return None
        
        rsi = calculate_rsi(data['Close'].values)
        if rsi is None or rsi > RSI_THRESHOLD:
            return None
        
        ma20 = calculate_ma(data['Close'].values, 20)
        ma50 = calculate_ma(data['Close'].values, 50)
        
        if ma20 is None or ma50 is None:
            return None
        
        if not (current_price > ma20 and current_price > ma50):
            return None
        
        if not check_last_5_support(data):
            return None
        
        if not check_4h_breakout(ticker):
            return None
        
        if not check_volatility(data):
            return None
        
        news_sentiment = check_news(ticker)
        if news_sentiment == "سلبية":
            return None
        
        short_ratio = get_short_ratio(ticker)
        if short_ratio and short_ratio > MAX_SHORT_RATIO:
            return None
        
        return {
            "ticker": ticker,
            "price": current_price,
            "rsi": rsi,
            "ma20": ma20,
            "ma50": ma50,
            "volume": volume,
            "news": news_sentiment,
            "short": short_ratio,
            "support": data.iloc[-6]['Low']
        }
        
    except Exception as e:
        print(f"❌ خطأ في تحليل {ticker}: {e}", flush=True)
        return None

def send_alert(result):
    msg = (
        f"<b>━━━━━━━━━━━━━━━━━━━━━</b>\n"
        f"<b>🎯 سهم جاهز للدخول!</b>\n"
        f"<b>━━━━━━━━━━━━━━━━━━━━━</b>\n\n"
        f"<b>📌 اسم السهم:</b> {result['ticker']}\n"
        f"<b>💵 السعر الحالي:</b> {p(result['price'])}\n"
        f"<b>📊 الدعم:</b> {p(result['support'])}\n\n"
        f"<b>📈 المؤشرات:</b>\n"
        f"🔴 RSI: {result['rsi']:.1f}\n"
        f"🟢 MA20: {p(result['ma20'])}\n"
        f"🟢 MA50: {p(result['ma50'])}\n"
        f"📦 الحجم: {numfmt(result['volume'])}\n\n"
        f"<b>✅ الشروط المحققة:</b>\n"
        f"✓ السعر بين 0.10 - 5 دولار\n"
        f"✓ RSI < 30\n"
        f"✓ السعر > MA20 و MA50\n"
        f"✓ آخر 5 جلسات لم تكسر الدعم\n"
        f"✓ فريم 4H صعود\n"
        f"✓ تذبذب ≤ 10%\n"
        f"✓ أخبار: {result['news'] or 'محايدة'}\n"
        f"<b>━━━━━━━━━━━━━━━━━━━━━</b>"
    )
    
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
        requests.post(url, json=payload, timeout=10)
        print(f"✅ تنبيه أرسل: {result['ticker']}", flush=True)
        return True
    except Exception as e:
        print(f"❌ خطأ إرسال: {e}", flush=True)
        return False

ALERTED = set()

def scan():
    print("=" * 50, flush=True)
    print("🚀 بدء فحص الأسهم...", flush=True)
    print("=" * 50, flush=True)
    
    try:
        print("📥 تحميل قائمة الأسهم...", flush=True)
        
        urls = [
            ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "Symbol"),
            ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "ACT Symbol")
        ]
        
        tickers = []
        for url, col in urls:
            try:
                df = pd.read_csv(url, sep="|")
                df = df[df["Test Issue"] == "N"]
                tickers.extend(df[col].dropna().astype(str).tolist())
            except:
                pass
        
        tickers = sorted(set(x.strip().upper() for x in tickers if x.strip().isalpha() and len(x.strip()) <= 5))
        
        print(f"✅ تم تحميل {len(tickers)} سهم", flush=True)
        
        found_count = 0
        
        for i in range(0, len(tickers), BATCH_SIZE):
            batch = tickers[i:i+BATCH_SIZE]
            print(f"📦 فحص {len(batch)} سهم...", flush=True)
            
            for ticker in batch:
                result = analyze_stock(ticker)
                
                if result:
                    found_count += 1
                    key = (result['ticker'], round(result['price'], 2))
                    
                    if key not in ALERTED:
                        if send_alert(result):
                            ALERTED.add(key)
                
                time.sleep(0.2)
            
            time.sleep(1)
        
        print("=" * 50, flush=True)
        print(f"✅ انتهى الفحص - وجدنا {found_count} إشارة", flush=True)
        print("=" * 50, flush=True)
        
    except Exception as e:
        print(f"❌ خطأ عام: {e}", flush=True)

if __name__ == "__main__":
    scan()
