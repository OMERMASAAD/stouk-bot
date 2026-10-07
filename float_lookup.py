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
import os
import threading
import time
import urllib.request

# SEC تحجب الطلبات بدون ترويسة تعريفية فيها بريد حقيقي — ضع بريدك في SEC_CONTACT (أو عدّل السطر)
SEC_CONTACT = os.environ.get("SEC_CONTACT", "your-email@example.com")
UA = "stouk-bot radar %s" % SEC_CONTACT
CACHE_FILE = os.environ.get("FLOAT_CACHE_FILE", "float_cache.json")
CACHE_TTL_DAYS = 21           # القيم الموجودة تُعاد استخدامها 21 يومًا
MISS_TTL_HOURS = 6            # الفشل يُعاد تجربته بعد 6 ساعات فقط
UNVERIFIED_CAP = 30_000_000   # أسهم مُصدَرة أكبر من هذا الرقم بلا Float دقيق = استبعاد
_cache = {}
_cache_dirty = False
_cache_lock = threading.Lock()
_sec_lock = threading.Lock()
_last_sec_call = [0.0]
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
_cik_attempts = 0
MAX_CIK_ATTEMPTS = 3          # لا نكرر محاولة تحميل ملف SEC أكثر من 3 مرات في التشغيل الواحد


def _http_json(url, timeout=15):
    if "sec.gov" in url:                      # حد SEC: 10 طلبات/ثانية — نلتزم بـ ~8
        with _sec_lock:
            wait = 0.125 - (time.time() - _last_sec_call[0])
            if wait > 0:
                time.sleep(wait)
            _last_sec_call[0] = time.time()
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
    """
    خريطة رمز السهم → رقم CIK من ملف SEC الرسمي (تُحمّل مرة واحدة).
    عند الفشل تُعاد المحاولة في النداء التالي (حتى 3 محاولات) بدل تعطيل SEC لكل الرموز.
    """
    global _cik_cache, _cik_attempts
    if _cik_cache is not None:
        return _cik_cache
    _cik_attempts += 1
    try:
        data = _http_json(SEC_TICKERS_URL)
        _cik_cache = {str(v.get("ticker", "")).upper(): int(v.get("cik_str"))
                      for v in data.values() if v.get("ticker") and v.get("cik_str")}
        print("SEC ticker map loaded:", len(_cik_cache), "tickers")
    except Exception as e:
        print("SEC cik map error (attempt %d/%d):" % (_cik_attempts, MAX_CIK_ATTEMPTS), e)
        if _cik_attempts >= MAX_CIK_ATTEMPTS:
            _cik_cache = {}
    return _cik_cache if _cik_cache is not None else {}


def sec_shares_outstanding(ticker):
    """
    أسهم قائمة المالكين من SEC EDGAR (حد أعلى للـ Float).
    يعيد (value, status) حيث status: used / no_cik / no_data / error.
    """
    cik = cik_map().get(str(ticker).upper())
    if not cik:
        return None, "no_cik"
    had_error = False
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
                return val, "used"
        except Exception as e:
            had_error = True
            print("SEC concept error", ticker, type(e).__name__, e)
            continue
    return None, ("error" if had_error else "no_data")


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


def load_cache():
    """تحميل float_cache.json (يُحفظ في المستودع فلا نعيد الاستعلام كل مرة)."""
    global _cache
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            _cache = json.load(f) or {}
    except Exception:
        _cache = {}
    return _cache


def save_cache():
    global _cache_dirty
    if not _cache_dirty:
        return
    with _cache_lock:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=0, sort_keys=True)
        _cache_dirty = False


def _cache_get(ticker):
    c = _cache.get(ticker)
    if not c:
        return None
    age_h = (time.time() - float(c.get("ts", 0))) / 3600.0
    ttl_h = MISS_TTL_HOURS if c.get("value") is None else CACHE_TTL_DAYS * 24
    return dict(c, source=c.get("source") or "cache") if age_h <= ttl_h else None


def _cache_put(ticker, res):
    global _cache_dirty
    with _cache_lock:
        _cache[ticker] = dict(res, ts=time.time())
        _cache_dirty = True


def _yahoo_chain(t):
    """أقصى ما يمكن من Yahoo بأقل طلبات: info ← fast_info ← get_shares_full."""
    float_shares = outstanding = None
    source = None
    try:
        info = t.info or {}
        if info.get("floatShares"):
            float_shares, source = int(info["floatShares"]), SOURCE_YAHOO_FLOAT
        v = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
        if v:
            outstanding = int(v)
            source = source or SOURCE_YAHOO_OUTSTANDING
    except Exception as e:
        print("float info error", t.ticker, type(e).__name__)
    if not outstanding and not float_shares:
        try:                                   # fast_info يستخدم نقطة نهاية مختلفة وينجح غالبًا عندما يفشل info
            v = t.fast_info.get("shares") if hasattr(t.fast_info, "get") else t.fast_info["shares"]
            if v:
                outstanding, source = int(v), SOURCE_YAHOO_OUTSTANDING
        except Exception:
            pass
    if not outstanding and not float_shares:
        try:
            s = t.get_shares_full(period="6mo")
            vals = [int(float(x)) for x in s.dropna() if float(x) > 0] if s is not None and len(s) else []
            if vals:
                outstanding, source = vals[-1], SOURCE_YAHOO_SHARES
        except Exception:
            pass
    return float_shares, outstanding, source


def resolve(ticker, mode="auto", use_cache=True):
    """
    يعيد {"value","exact","source","float_shares","shares_outstanding","sec_status"}.
    الترتيب: الكاش ← Yahoo (info/fast_info/shares_full) ← SEC EDGAR.
    Float الدقيق من Yahoo فقط؛ الباقي «حد أعلى» (أسهم مُصدَرة).
    النتائج (حتى الفاشلة) تُخزَّن في الكاش لتقليل الطلبات وتفادي حظر Yahoo.
    """
    import yfinance as yf
    if use_cache:
        c = _cache_get(ticker)
        if c is not None:
            return c
    float_shares, outstanding, source = _yahoo_chain(yf.Ticker(ticker))
    sec_status = "skipped"
    if mode != "strict" and not float_shares and not outstanding:
        sec, sec_status = sec_shares_outstanding(ticker)
        if sec:
            outstanding, source = sec, SOURCE_SEC
    if float_shares:
        res = {"value": float_shares, "exact": True, "source": source or SOURCE_YAHOO_FLOAT}
    elif outstanding and mode != "strict":
        res = {"value": outstanding, "exact": False, "source": source or SOURCE_YAHOO_OUTSTANDING}
    else:
        res = {"value": None, "exact": False, "source": None}
    res.update({"float_shares": float_shares, "shares_outstanding": outstanding, "sec_status": sec_status})
    _cache_put(ticker, res)
    return res


def classify(info, max_float, policy="watch", cap=UNVERIFIED_CAP):
    """
    يصنّف الـ Float ويعيد (accepted, status):
      exact      : Float دقيق ≤ الحد                      → مقبول
      bound      : لا Float دقيق لكن الأسهم المُصدَرة ≤ الحد (Float ≤ ذلك قطعًا) → مقبول
      unverified : لا Float دقيق والأسهم المُصدَرة بين الحد و cap → مقبول بتحفظ (policy=watch)
      unknown    : لا توجد أي بيانات أسهم                   → مقبول بتحفظ (policy=watch)
      too_big    : Float دقيق أكبر من الحد أو أسهم مُصدَرة > cap → مرفوض
    policy="exclude" يعيد السلوك القديم الصارم (يقبل فقط value ≤ max_float).
    """
    info = info or {}
    value, exact = info.get("value"), bool(info.get("exact"))
    if policy == "exclude":
        ok = value is not None and value <= max_float
        return ok, ("exact" if ok and exact else "bound" if ok else "too_big" if value is not None else "unknown")
    if value is None:
        return True, "unknown"
    if value <= max_float:
        return True, ("exact" if exact else "bound")
    if exact:
        return False, "too_big"
    return (True, "unverified") if value <= cap else (False, "too_big")


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
