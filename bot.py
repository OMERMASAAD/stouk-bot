"""
بوت قنص الزخم اللحظي - نسخة محسّنة (v2)
التحديثات:
- MIN_GAIN_PCT رُفعت لـ 20% (بدل 10%) لاستبعاد الأسهم الخاملة
- فلتر "حداثة الحركة": يتأكد إن السعر لسه قريب من آخر قاع بآخر ساعة، مو حركة ممتدة خلصت
- معالجة أخطاء أقوى: أي خطأ بدفعة وحدة ما يوقف باقي الفحص كامل (يفسر التشغيلات اللي فشلت بسرعة)
- طباعة تفصيلية أكثر بالـ log عشان لو صار خطأ نقدر نشخصه بسهولة من صفحة Actions
"""

import os
import sys
import time
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN") or "ضع_التوكن_هنا"
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID") or "ضع_الشات_آيدي_هنا"

MIN_PRICE = 0.10
MAX_PRICE = 3.00
MIN_GAIN_PCT = 20                  # رُفعت من 10 إلى 20 (حسب الطلب)
MIN_RELATIVE_VOLUME = 3.0
MIN_DOLLAR_LIQUIDITY = 500_000      # أقل سيولة دولارية مقبولة (نصف مليون$) عشان تضمن دخول وخروج سهل
MAX_EXTENSION_FROM_RECENT_LOW_PCT = 15  # السعر ما يتجاوز 15% فوق أقل قاع بآخر 3 شمعات (حركة طازة)
RECENT_LOW_LOOKBACK_CANDLES = 3     # عدد شمعات الـ15 دقيقة نرجع لها (~45 دقيقة)
CHUNK_SIZE = 150

alerted_today = set()
current_day = datetime.now().date()


def get_all_tickers():
    tickers = []
    try:
        nasdaq = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", sep="|")
        nasdaq = nasdaq[nasdaq["Test Issue"] == "N"]
        tickers += nasdaq["Symbol"].dropna().tolist()
    except Exception as e:
        print("خطأ NASDAQ:", e, flush=True)
    try:
        other = pd.read_csv("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", sep="|")
        other = other[other["Test Issue"] == "N"]
        tickers += other["ACT Symbol"].dropna().tolist()
    except Exception as e:
        print("خطأ NYSE/AMEX:", e, flush=True)
    return sorted(set(t for t in tickers if isinstance(t, str) and t.isalpha() and len(t) <= 5))


def send_telegram_alert(ticker, price, gain_pct, rel_vol, dollar_liquidity, shares_outstanding, market_cap, extension_pct):
    message = (
        f"🔴 تنبيه اختراق: {ticker}\n"
        f"نسبة الارتفاع (من فتح اليوم) -> {gain_pct:.0f}%\n"
        f"الامتداد عن آخر قاع -> {extension_pct:.0f}% (حركة طازة)\n"
        f"السعر -> {price:.2f} دولار\n"
        f"القيمة السوقية -> {market_cap/1_000_000:.1f} مليون\n"
        f"الحجم النسبي -> {rel_vol:.0f}X مرة\n"
        f"السيولة الحالية -> {dollar_liquidity/1_000_000:.2f}M$\n"
        f"الأسهم المصدرة -> {shares_outstanding/1_000_000:.1f} مليون\n"
        f"⚠️ الأسعار المجانية متأخرة 15-20 دقيقة عن اللحظي، تحقق من الشارت الفعلي قبل القرار."
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15)
        print("تم إرسال تنبيه:", ticker, flush=True)
    except Exception as e:
        print("فشل الإرسال:", e, flush=True)


def scan_chunk(tickers_chunk, chunk_num, total_chunks):
    global alerted_today
    print(f"[دفعة {chunk_num}/{total_chunks}] جاري تحميل بيانات {len(tickers_chunk)} سهم...", flush=True)
    try:
        data = yf.download(
            tickers=tickers_chunk, period="2d", interval="15m",
            group_by="ticker", threads=True, progress=False,
        )
    except Exception as e:
        print(f"[دفعة {chunk_num}] خطأ بتحميل الدفعة كاملة (تجاهلناها وكملنا): {e}", flush=True)
        return

    for ticker in tickers_chunk:
        try:
            df = data[ticker] if len(tickers_chunk) > 1 else data
            df = df.dropna()
            if df.empty or len(df) < RECENT_LOW_LOOKBACK_CANDLES + 5:
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

            # فلتر حداثة الحركة: السعر الحالي ما يكون بعيد كثير عن أقرب قاع بآخر عدة شمعات
            recent_low = df["Low"].tail(RECENT_LOW_LOOKBACK_CANDLES).min()
            if recent_low <= 0:
                continue
            extension_pct = ((last_close - recent_low) / recent_low) * 100
            if extension_pct > MAX_EXTENSION_FROM_RECENT_LOW_PCT:
                continue  # الحركة خلصت أوجها، السهم "طار" بعيد عن القاع الأخير

            avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
            last_vol = df["Volume"].iloc[-1]
            if avg_vol == 0 or pd.isna(avg_vol):
                continue
            rel_vol = last_vol / avg_vol
            if rel_vol < MIN_RELATIVE_VOLUME:
                continue

            today_volume_sum = df["Volume"].tail(26).sum()
            dollar_liquidity = today_volume_sum * last_close
            if dollar_liquidity < MIN_DOLLAR_LIQUIDITY:
                continue  # سيولة ضعيفة، السهم يصعب الدخول والخروج منه بأمان

            tk = yf.Ticker(ticker)
            info = tk.info
            shares_outstanding = info.get("sharesOutstanding") or info.get("floatShares") or 0
            market_cap = info.get("marketCap") or (shares_outstanding * last_close)

            if ticker not in alerted_today:
                send_telegram_alert(
                    ticker, last_close, gain_pct, rel_vol,
                    dollar_liquidity, shares_outstanding, market_cap, extension_pct,
                )
                alerted_today.add(ticker)

        except Exception as e:
            print(f"[{ticker}] خطأ فردي (تجاهلناه وكملنا): {e}", flush=True)
            continue


def main():
    global current_day, alerted_today
    print("جاري تجهيز قائمة الأسهم...", flush=True)
    try:
        all_tickers = get_all_tickers()
    except Exception as e:
        print("خطأ فادح بجلب قائمة الأسهم:", e, flush=True)
        sys.exit(1)

    if not all_tickers:
        print("تحذير: قائمة الأسهم فاضية، توقف بدون فحص.", flush=True)
        sys.exit(1)

    print(f"تم تحميل {len(all_tickers)} سهم. بدء الفحص.", flush=True)

    if datetime.now().date() != current_day:
        alerted_today = set()
        current_day = datetime.now().date()

    total_chunks = (len(all_tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE
    start = time.time()
    for idx, i in enumerate(range(0, len(all_tickers), CHUNK_SIZE), start=1):
        scan_chunk(all_tickers[i:i + CHUNK_SIZE], idx, total_chunks)

    elapsed = time.time() - start
    print(f"انتهى الفحص خلال {elapsed:.1f} ثانية.", flush=True)


if __name__ == "__main__":
    main()
