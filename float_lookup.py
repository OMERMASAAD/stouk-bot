# -*- coding: utf-8 -*-
"""
جلب Float / عدد الأسهم المُصدَرة من مصادر مجانية (بدون مفاتيح).

الأولوية:
  1) Yahoo Finance: floatShares          → Float دقيق.
  2) Yahoo Finance: sharesOutstanding    → حد أعلى للـ Float (الأسهم المُصدَرة).
  3) yfinance: get_shares_full / balance_sheet → حد أعلى.
  4) SEC EDGAR (XBRL): أسهم قائمة المالكين المسجّلة → حد أعلى.

القاعدة: Float ≤ Shares Outstanding دائمًا، لذلك إذا كانت الأسهم المُصدَرة ≤ 10M فالسهم
منخفض الـ Float بالضرورة (قبول آمن)، وإذا كانت أكبر من 10M لا يمكن الجزم فيُستبعد
وفق الشرط الصارم. الدقة تُعلَن عبر `exact` و`source`.

ملاحظة SEC: تتطلب ترويسة User-Agent تعرّف بالجهة المستعلِمة، ومعدل 10 طلبات/ثانية كحد أقصى.
"""
import json
import urllib.request

UA = "stouk-bot radar (github.com/OMERMASAAD/stouk-bot)"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_CONCEPT_URLS = [
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/dei/EntityCommonStockSharesOutstanding.json",
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/CommonStockSharesOutstanding.json",
    "https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/WeightedAverageNumberOfSharesOutstandingBasic.json",
]
SOURCE_YAHOO_FLOAT = "yahoo_float"
SOURCE_YAHOO_OUTSTANDING = "yahoo_outstanding"
SOURCE_YAHOO_SHARES = "yahoo_shares_full"
SOURCE_SEC = "sec_edgar"

_cik_cache = None


def _http_json(url, timeout=12):
    req = urllib.request.Request(url, headers={
        "User-Agent": UA, "Accept": "application/json", "Accept-Encoding": "gzip, deflate",
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        if resp.headers.get("Content-Encoding") == "gzip":
            import gzip
            raw = gzip.decompress(raw)
        return json.loads(raw.decode("utf-8", "replace"))


def cik_map():
    """خريطة رمز السهم → رقم CIK من ملف SEC الرسمي (تُحمّل مرة واحدة)."""
    global _cik_cache
    if _cik_cache is None:
        try:
            data = _http_json(SEC_TICKERS_URL)
            _cik_cache = {str(v.get("ticker", "")).upper(): int(v.get("cik_str"))
                          for v in data.values() if v.get("ticker") and v.get("cik_str")}
        except Exception as e:
            print("SEC cik map error:", e)
            _cik_cache = {}
    return _cik_cache


def sec_shares_outstanding(ticker):
    """أسهم قائمة المالكين من SEC EDGAR (حد أعلى للـ Float) أو None."""
    cik = cik_map().get(str(ticker).upper())
    if not cik:
        return None
    for tpl in SEC_CONCEPT_URLS:
        try:
            d = _http_json(tpl.format(cik=f"{cik:010d}"))
            units = (d.get("units") or {}).get("shares") or []
            rows = [u for u in units if u.get("val") is not None]
            if not rows:
                continue
            rows.sort(key=lambda u: (str(u.get("end") or ""), str(u.get("filed") or "")))
            val = int(rows[-1]["val"])
            if val > 0:
                return val
        except Exception:
            continue
    return None


def yahoo_shares_outstanding(ticker):
    """sharesOutstanding من yfinance info (حد أعلى) أو None."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        v = info.get("sharesOutstanding")
        return int(v) if v else None
    except Exception as e:
        print("sharesOutstanding error", ticker, e)
        return None


def resolve(ticker, mode="auto"):
    """
    يعيد {"value", "exact", "source", "float_shares", "shares_outstanding"}.
    mode="auto"   : يُقبل الحد الأعلى (شرط: ≤ 10M في مرحلة الفلترة).
    mode="strict" : Float الدقيق فقط، وما عداه غير متوفر.
    """
    import yfinance as yf

    float_shares = None
    outstanding = None
    source = None
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
        f = info.get("floatShares")
        if f:
            float_shares = int(f)
            source = SOURCE_YAHOO_FLOAT
        if not outstanding and info.get("sharesOutstanding"):
            outstanding = int(info["sharesOutstanding"])
            source = source or SOURCE_YAHOO_OUTSTANDING
    except Exception as e:
        print("float info error", ticker, e)

    if not outstanding:
        try:
            s = t.get_shares_full(period="6mo")
            if s is not None and len(s):
                vals = [int(float(x)) for x in s.dropna() if float(x) > 0]
                if vals:
                    outstanding = vals[-1]
                    source = source or SOURCE_YAHOO_SHARES
        except Exception as e:
            print("shares_full error", ticker, e)

    if not outstanding:
        try:
            bs = t.balance_sheet
            if bs is not None and not bs.empty:
                for key in ("Ordinary Shares Number", "Share Issued"):
                    if key in bs.index:
                        vals = [int(float(x)) for x in bs.loc[key].dropna() if float(x) > 0]
                        if vals:
                            outstanding = vals[0]
                            source = source or SOURCE_YAHOO_SHARES
                            break
        except Exception as e:
            print("balance sheet error", ticker, e)

    if not outstanding and mode == "auto":
        sec = sec_shares_outstanding(ticker)
        if sec:
            outstanding = sec
            source = SOURCE_SEC

    if float_shares:
        return {"value": float_shares, "exact": True, "source": source or SOURCE_YAHOO_FLOAT,
                "float_shares": float_shares, "shares_outstanding": outstanding}
    if outstanding and mode != "strict":
        return {"value": outstanding, "exact": False, "source": source or SOURCE_YAHOO_OUTSTANDING,
                "float_shares": None, "shares_outstanding": outstanding}
    return {"value": None, "exact": False, "source": None,
            "float_shares": float_shares, "shares_outstanding": outstanding}


def label(value, exact):
    """تسمية الـ Float للعرض: دقيق «4.2M» أو حد أعلى «≤ 8.3M»."""
    if value is None:
        return "غير متوفر"
    try:
        v = float(value)
    except Exception:
        return "غير متوفر"
    txt = f"{v / 1_000_000:.1f}M" if v >= 1_000_000 else (f"{v / 1_000:.0f}K" if v >= 1_000 else str(int(v)))
    return txt if exact else f"≤ {txt}"
