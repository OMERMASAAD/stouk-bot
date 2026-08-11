"""
بوت استراتيجية "التجميع" - يعتمد على شمعة فوليوم ضخمة (50 مليون+) كأول إشارة تجميع،
ثم يراقب الارتداد لمنطقة الشراء (فوق سعر الافتتاح وحتى 20% فوقه) خلال الأيام التالية.

هذي استراتيجية "بطيئة" (أيام مو دقائق)، فالفحص مرتين باليوم كافي ومناسب.

الشروط المطبقة:
1) السعر بين 1$ و4$
2) حجم تداول الشمعة اليومية 50 مليون سهم+
3) "أول شمعة فقط" - ما فيه شمعة سابقة بآخر 20 جلسة وصلت نفس الحجم (يستبعد الأسهم
   اللي أصلاً حجمها عالي دايماً زي الشركات الكبيرة)
4) الإشارة لازم تكون حديثة (خلال آخر 15 جلسة) لسه فعالة
5) السعر الحالي داخل منطقة الشراء (من سعر الافتتاح وحتى 20% فوقه)
6) وقف الخسارة (السعر الحالي ما كسر 15% تحت سعر افتتاح شمعة الإشارة) لسه ما انكسر
7) فلوت أكبر من 100 مليون سهم (الاستراتيجية مصممة لأسهم فلوت أكبر، مو بيني ستوك صغير)

ملاحظة: نسبة النجاح المذكورة بمصدر الاستراتيجية (95%) ادعاء غير موثق، والبوت هنا
بس يطبق المنطق الفني للفلترة، مو ضمان لأي نتيجة.
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

MIN_PRICE = 1.00
MAX_PRICE = 4.00
MIN_DAILY_VOLUME_SHARES = 50_000_000
FRESHNESS_LOOKBACK_DAYS = 20      # ما فيه شمعة سابقة بنفس الحجم خلال هالمدة
SIGNAL_RECENCY_DAYS = 15          # الإشارة لازم تكون خلال آخر كم جلسة
BUY_ZONE_UPPER_MULTIPLIER = 1.20  # سقف منطقة الشراء = 20% فوق الافتتاح
STOP_LOSS_MULTIPLIER = 0.85       # وقف الخسارة = 15% تحت الافتتاح
TARGET_MULTIPLIER = 2.0           # الهدف المتوقع = ضعف منطقة الشراء العليا
MIN_FLOAT = 100_000_000           # الاستراتيجية لأسهم فلوت أكبر، مو بيني ستوك صغير جداً

CHUNK_SIZE = 100
alerted_signals = set()  # (ticker, signal_date) عشان ما نكرر نفس الإشارة


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


def send_telegram_alert(ticker, signal_date, signal_open, buy_low, buy_high, stop_loss,
                         target, current_price, float_shares, signal_volume):
    message = (
        f"📊 فرصة تجميع: {ticker}\n"
        f"تاريخ شمعة الفوليوم -> {signal_date}\n"
        f"حجم شمعة الإشارة -> {signal_volume/1_000_000:.1f} مليون سهم\n"
        f"السعر الحالي -> {current_price:.2f} دولار\n"
        f"منطقة الشراء المقترحة -> ${buy_low:.2f} - ${buy_high:.2f} (على دفعات، مو دفعة وحدة)\n"
        f"وقف الخسارة -> ${stop_loss:.2f}\n"
        f"الهدف المتوقع -> ${target:.2f}\n"
        f"الفلوت -> {float_shares/1_000_000:.1f} مليون\n"
        f"مدة الاحتفاظ المتوقعة -> يوم إلى أسبوعين حسب حركة السهم\n"
        f"⚠️ هذي استراتيجية أيام مو دقائق، التأخير هنا مو مؤثر. راجع الأخبار والشارت بنفسك، مو توصية."
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=15)
        print("تم إرسال تنبيه:", ticker, flush=True)
    except Exception as e:
        print("فشل الإرسال:", e, flush=True)


def scan_chunk(tickers_chunk, chunk_num, total_chunks):
    global alerted_signals
    print(f"[دفعة {chunk_num}/{total_chunks}] جاري تحميل بيانات {len(tickers_chunk)} سهم...", flush=True)
    try:
        data = yf.download(
            tickers=tickers_chunk, period="90d", interval="1d",
            group_by="ticker", threads=True, progress=False,
        )
    except Exception as e:
        print(f"[دفعة {chunk_num}] خطأ بتحميل الدفعة (تجاهلناها وكملنا): {e}", flush=True)
        return

    for ticker in tickers_chunk:
        try:
            df = data[ticker] if len(tickers_chunk) > 1 else data
            df = df.dropna()
            if df.empty or len(df) < FRESHNESS_LOOKBACK_DAYS + SIGNAL_RECENCY_DAYS:
                continue

            current_price = df["Close"].iloc[-1]
            if not (MIN_PRICE <= current_price <= MAX_PRICE):
                continue

            # نبحث عن أحدث "شمعة إشارة" (فوليوم 50 مليون+) خلال آخر SIGNAL_RECENCY_DAYS جلسة
            recent_window = df.tail(SIGNAL_RECENCY_DAYS)
            signal_candidates = recent_window[recent_window["Volume"] >= MIN_DAILY_VOLUME_SHARES]
            if signal_candidates.empty:
                continue

            # نأخذ أقدم إشارة بالنطاق الحديث (أقرب لبداية الحركة، مو آخر يوم صعد فيه الفوليوم)
            signal_idx = signal_candidates.index[0]
            signal_pos = df.index.get_loc(signal_idx)

            # شرط "أول شمعة": ما فيه فوليوم مشابه بآخر FRESHNESS_LOOKBACK_DAYS يوم قبلها
            lookback_start = max(0, signal_pos - FRESHNESS_LOOKBACK_DAYS)
            prior_window = df.iloc[lookback_start:signal_pos]
            if not prior_window.empty and (prior_window["Volume"] >= MIN_DAILY_VOLUME_SHARES).any():
                continue  # فيه فوليوم مشابه سابق، مو أول شمعة فعلية

            signal_row = df.loc[signal_idx]
            signal_open = signal_row["Open"]
            signal_volume = signal_row["Volume"]
            if signal_open <= 0:
                continue

            buy_low = signal_open
            buy_high = signal_open * BUY_ZONE_UPPER_MULTIPLIER
            stop_loss = signal_open * STOP_LOSS_MULTIPLIER
            target = buy_high * TARGET_MULTIPLIER

            # السعر الحالي لازم يكون داخل منطقة الشراء
            if not (buy_low <= current_price <= buy_high):
                continue

            # وقف الخسارة ما ينكسر: نتأكد ما فيه إغلاق تحته من يوم الإشارة لين الحين
            after_signal = df.loc[signal_idx:]
            if (after_signal["Close"] < stop_loss).any():
                continue  # الوقف انكسر سابقاً، الفرصة انتهت

            tk = yf.Ticker(ticker)
            info = tk.info
            float_shares = info.get("floatShares") or 0
            if float_shares < MIN_FLOAT:
                continue

            signal_date_str = str(signal_idx.date())
            key = (ticker, signal_date_str)
            if key not in alerted_signals:
                send_telegram_alert(
                    ticker, signal_date_str, signal_open, buy_low, buy_high,
                    stop_loss, target, current_price, float_shares, signal_volume,
                )
                alerted_signals.add(key)

        except Exception as e:
            print(f"[{ticker}] خطأ فردي (تجاهلناه وكملنا): {e}", flush=True)
            continue


def main():
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

    total_chunks = (len(all_tickers) + CHUNK_SIZE - 1) // CHUNK_SIZE
    start = time.time()
    for idx, i in enumerate(range(0, len(all_tickers), CHUNK_SIZE), start=1):
        scan_chunk(all_tickers[i:i + CHUNK_SIZE], idx, total_chunks)

    elapsed = time.time() - start
    print(f"انتهى الفحص خلال {elapsed:.1f} ثانية.", flush=True)


if __name__ == "__main__":
    main()
