"""
بوت قنص الزخم اللحظي - نسخة GitHub Actions
يفحص عينة من الأسهم منخفضة السعر، ويبعث تنبيه تيليجرام عند تحقق الشروط
"""

import time
import os
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

MIN_PRICE = 0.10
MAX_PRICE = 3.00
MIN_GAIN_PCT = 10
MIN_RELATIVE_VOLUME = 3.0
CHUNK_SIZE = 150


def get_all_tickers():
    tickers = []
    try:
        nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", sep="|")
        nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]
        tickers += nasdaq["Symbol"].dropna().tolist()
    except Exception as e:
        print("خطأ NASDAQ:", e)
    try:
        other = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", sep="|")
        other = other[other["Test Issue"] == "N"]
        tickers += other["ACT Symbol"].dropna().tolist()
    except Exception as e:
        print("خطأ NYSE/AMEX:", e)
    return sorted(set(t for t in tickers if isinstance(t, str) and t.isalpha() and len(t) <= 5))


def send_telegram_alert(ticker, price, gain_pct, rel_vol, dollar_liquidity, shares_outstanding, market_cap):
    message = (
        f"🔴 تنبيه اختراق: {ticker}\n"
        f"نسبة الارتفاع -> {gain_pct:.0f}%\n"
        f"السعر -> {price:.2f} دولار\n"
        f"القيمة السوقية -> {market_cap/1_000_000:.1f} مليون\n"
        f"الحجم النسبي -> {rel_vol:.0f}X مرة\n"
        f"السيولة الحالية -> {dollar_liquidity/1_000_000:.2f}M$\n"
        f"الأسهم المتداولة -> {shares_outstanding/1_000_000:.1f} مليون\n"
        f"⚠️ تنبيه فني فقط، راجع الشارت والأخبار قبل أي قرار."
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        print("تم إرسال تنبيه:", ticker)
    except Exception as e:
        print("فشل الإرسال:", e)


def scan_chunk(tickers_chunk):
    try:
        data = yf.download(tickers=tickers_chunk, period="2d", interval="15m", group_by="ticker", threads=True, progress=False)
    except Exception as e:
        print("خطأ بتحميل دفعة:", e)
        return

    for ticker in tickers_chunk:
        try:
            df = data[ticker] if len(tickers_chunk) > 1 else data
            df = df.dropna()
            if df.empty or len(df) < 20:
                continue

            last_close = df["Close"].iloc[-1]
            if not (MIN_PRICE <= last_close <= MAX_PRICE):
                continue

            day_open = df["Open"].iloc[0]
            if day_open <= 0:
                continue
            gain_pct = ((last_close - day_open) / day_open) * 100
            if gain_pct < MIN_GAIN_PCT:
                continue

            avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
            last_vol = df["Volume"].iloc[-1]
            if avg_vol == 0 or pd.isna(avg_vol):
                continue
            
            rel_vol = last_vol / avg_vol
            if rel_vol < MIN_RELATIVE_VOLUME:
                continue

            today_volume_sum = df["Volume"].tail(26).sum()
            dollar_liquidity = today_volume_sum * last_close

            tk = yf.Ticker(ticker)
            info = tk.info
            shares_outstanding = info.get("sharesOutstanding") or info.get("floatShares") or 0
            market_cap = info.get("marketCap") or (shares_outstanding * last_close)

            send_telegram_alert(ticker, last_close, gain_pct, rel_vol, dollar_liquidity, shares_outstanding, market_cap)

        except Exception:
            continue


def main():
    print("جاري تجهيز قائمة الأسهم...")
    all_tickers = get_all_tickers()
    print(f"تم تحميل {len(all_tickers)} سهم. بدء الفحص...")

    for i in range(0, len(all_tickers), CHUNK_SIZE):
        scan_chunk(all_tickers[i:i + CHUNK_SIZE])

    print("انتهى الفحص.")


if __name__ == "__main__":
    main()
