"""
بوت قنص الزخم اللحظي - نسخة v3 (استراتيجية Penny Stocks & Low-Float الاحترافية)

أولاً - تصفية الكون (Universe Filtering):
  - السعر 0.10$ - 3.00$
  - حجم التداول اليومي >= 1,000,000 سهم
  - الحجم النسبي (RVol) >= 3
  - الفلوت <= 50 مليون سهم

ثانياً - شروط الدخول اللحظية (لازم تتحقق كلها بنفس اللحظة):
  - اختراق أعلى سعر باليوم الحالي (Breakout)
  - EMA9 فوق EMA20 وترتفع (تقاطع إيجابي قصير المدى)
  - RSI(14) بين 60 و 80
  - السعر فوق VWAP
  - شمعة انفجار: حجم عالي + جسم قوي + إغلاق قريب من القمة

ثالثاً - الأمان:
  - تجنب الأسهم المرتفعة أكثر من 100% باليوم (فخ التصريف)
  - فلتر حداثة الحركة (نفس فلتر النسخة السابقة)
  - وقف خسارة ابتدائي مقترح فقط (البوت لا يدير صفقة مفتوحة أو وقف متحرك حي،
    هذا يحتاج بوت تداول فعلي منفصل يتابع الصفقة لحظياً)
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

# ==================== أولاً: تصفية الكون ====================
MIN_PRICE = 0.10
MAX_PRICE = 3.00
MIN_DAILY_VOLUME_SHARES = 1_000_000
MIN_RELATIVE_VOLUME = 3.0
MAX_FLOAT = 50_000_000

# ==================== ثانياً: شروط الدخول ====================
RSI_LOW = 60
RSI_HIGH = 80
MIN_BODY_RATIO = 0.6        # جسم الشمعة لازم يكون 60%+ من مدى الشمعة كاملة
MAX_UPPER_WICK_RATIO = 0.3  # الإغلاق قريب من القمة (ذيل علوي صغير)

# ==================== ثالثاً: الأمان ====================
MAX_DAY_GAIN_PCT = 100                  # تجنب الأسهم المتضخمة فوق 100% (فخ تصريف)
MAX_EXTENSION_FROM_RECENT_LOW_PCT = 15  # فلتر حداثة الحركة
RECENT_LOW_LOOKBACK_CANDLES = 3
STOP_LOSS_PCT = 0.04                    # وقف خسارة ابتدائي مقترح 4% تحت الدخول

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


def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def compute_vwap(df_today):
    typical_price = (df_today["High"] + df_today["Low"] + df_today["Close"]) / 3
    cum_pv = (typical_price * df_today["Volume"]).cumsum()
    cum_vol = df_today["Volume"].cumsum()
    return cum_pv / cum_vol


def send_telegram_alert(ticker, price, gain_pct, rel_vol, dollar_liquidity, shares_outstanding,
                         market_cap, extension_pct, rsi_val, float_shares, stop_loss):
    message = (
        f"🚀 تنبيه دخول قوي: {ticker}\n"
        f"نسبة الارتفاع (من فتح اليوم) -> {gain_pct:.0f}%\n"
        f"الامتداد عن آخر قاع -> {extension_pct:.0f}%\n"
        f"السعر -> {price:.2f} دولار\n"
        f"RSI(14) -> {rsi_val:.0f} | فوق EMA9/EMA20 و VWAP ✅\n"
        f"القيمة السوقية -> {market_cap/1_000_000:.1f} مليون\n"
        f"الحجم النسبي -> {rel_vol:.0f}X مرة\n"
        f"السيولة الحالية -> {dollar_liquidity/1_000_000:.2f}M$\n"
        f"الفلوت -> {float_shares/1_000_000:.1f} مليون\n"
        f"وقف خسارة ابتدائي مقترح -> ${stop_loss:.2f} (مو وقف متحرك حي، راقبه بنفسك)\n"
        f"⚠️ الأسعار المجانية متأخرة 15-20 دقيقة، تحقق من الشارت الفعلي قبل القرار."
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
            if df.empty or len(df) < 25:
                continue

            # نحصر بيانات اليوم الحالي بس (لحساب فتح اليوم، أعلى قمة، VWAP)
            last_date = df.index[-1].date()
            df_today = df[df.index.date == last_date]
            if len(df_today) < 5:
                continue  # لسه بدري باليوم، ما فيه بيانات كافية

            last_close = df_today["Close"].iloc[-1]
            if not (MIN_PRICE <= last_close <= MAX_PRICE):
                continue

            day_open = df_today["Open"].iloc[0]
            if day_open <= 0:
                continue
            gain_pct = ((last_close - day_open) / day_open) * 100
            if gain_pct > MAX_DAY_GAIN_PCT:
                continue  # فخ تصريف محتمل، تجاهله

            # اختراق أعلى سعر باليوم (باستثناء الشمعة الحالية)
            prior_bars = df_today.iloc[:-1]
            if prior_bars.empty:
                continue
            day_high_so_far = prior_bars["High"].max()
            if last_close <= day_high_so_far:
                continue  # ما اخترق أعلى قمة سابقة باليوم

            # فلتر حداثة الحركة
            recent_low = df["Low"].tail(RECENT_LOW_LOOKBACK_CANDLES).min()
            if recent_low <= 0:
                continue
            extension_pct = ((last_close - recent_low) / recent_low) * 100
            if extension_pct > MAX_EXTENSION_FROM_RECENT_LOW_PCT:
                continue

            # الحجم النسبي
            avg_vol = df["Volume"].rolling(20).mean().iloc[-1]
            last_vol = df["Volume"].iloc[-1]
            if avg_vol == 0 or pd.isna(avg_vol):
                continue
            rel_vol = last_vol / avg_vol
            if rel_vol < MIN_RELATIVE_VOLUME:
                continue

            # حجم التداول اليومي بالأسهم
            daily_volume_shares = df_today["Volume"].sum()
            if daily_volume_shares < MIN_DAILY_VOLUME_SHARES:
                continue

            # EMA9 / EMA20
            ema9 = compute_ema(df["Close"], 9)
            ema20 = compute_ema(df["Close"], 20)
            if pd.isna(ema9.iloc[-1]) or pd.isna(ema20.iloc[-1]) or pd.isna(ema9.iloc[-2]):
                continue
            if not (ema9.iloc[-1] > ema20.iloc[-1] and ema9.iloc[-1] > ema9.iloc[-2]):
                continue  # EMA9 لازم يكون فوق EMA20 ومرتفع عن الشمعة اللي قبلها

            # RSI(14)
            rsi_series = compute_rsi(df["Close"])
            last_rsi = rsi_series.iloc[-1]
            if pd.isna(last_rsi) or not (RSI_LOW <= last_rsi <= RSI_HIGH):
                continue

            # VWAP
            vwap_series = compute_vwap(df_today)
            last_vwap = vwap_series.iloc[-1]
            if pd.isna(last_vwap) or last_close <= last_vwap:
                continue

            # شمعة انفجار: جسم قوي + إغلاق قريب من القمة
            last_bar = df_today.iloc[-1]
            candle_range = last_bar["High"] - last_bar["Low"]
            if candle_range <= 0:
                continue
            body_ratio = abs(last_bar["Close"] - last_bar["Open"]) / candle_range
            upper_wick_ratio = (last_bar["High"] - last_bar["Close"]) / candle_range
            if body_ratio < MIN_BODY_RATIO or upper_wick_ratio > MAX_UPPER_WICK_RATIO:
                continue

            # سيولة وفلوت
            today_volume_sum = df["Volume"].tail(26).sum()
            dollar_liquidity = today_volume_sum * last_close

            tk = yf.Ticker(ticker)
            info = tk.info
            float_shares = info.get("floatShares") or 0
            if float_shares == 0 or float_shares > MAX_FLOAT:
                continue

            shares_outstanding = info.get("sharesOutstanding") or float_shares
            market_cap = info.get("marketCap") or (shares_outstanding * last_close)

            stop_loss = last_close * (1 - STOP_LOSS_PCT)

            if ticker not in alerted_today:
                send_telegram_alert(
                    ticker, last_close, gain_pct, rel_vol, dollar_liquidity,
                    shares_outstanding, market_cap, extension_pct, last_rsi,
                    float_shares, stop_loss,
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
