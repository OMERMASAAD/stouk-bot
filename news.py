# -*- coding: utf-8 -*-
"""
طبقة الأخبار والأحداث — مصدرها Yahoo Finance فقط:
  - عناوين الأخبار: yfinance Ticker.news
  - مواعيد النتائج المالية (مستقبلية): yfinance Ticker.calendar

حدود Yahoo: لا يوجد فيه وثائق SEC ولا صفحات Investor Relations،
والتواريخ المستقبلية المؤكدة المتاحة هي موعد النتائج المالية فقط.
أما بقية المحفزات فتُستخرج من عناوين الأخبار (الأحدث منها) وتُعرض بتاريخ نشر الخبر.
"""
import re
from datetime import datetime, timezone

# ---- أحداث سلبية (تخفيف/تمويل) ----
NEGATIVE = [
    (r"\bat[- ]the[- ]market\b|\bATM (offering|program|facility)\b", "ATM Offering"),
    (r"\bdirect (registered )?offering\b|\bregistered direct\b", "Direct Offering"),
    (r"\bpublic offering\b|\bunderwritten offering\b|\bstock offering\b|\bshare offering\b|\boffering of\b|\bproposed offering\b|\bpricing of\b.*\boffering\b", "Public/Stock Offering"),
    (r"\bprivate placement\b|\bPIPE\b", "Private Placement"),
    (r"\bdilut\w*", "Dilution"),
    (r"\bwarrants?\b", "Warrants"),
    (r"\bconvertible (senior )?notes?\b|\bconvertible debenture", "Convertible Notes"),
    (r"\bregistration statement\b|\bshelf (registration|offering)\b|\bform S-[13]\b|\bS-3\b", "Registration Statement"),
    (r"\bsells? (shares|stock)\b|\bshare sale\b|\bequity (line|financing|raise)\b|\braises? \$", "Share Sale / Financing"),
]
# كلمات تدل أن الحدث تم فعلًا
ALREADY = re.compile(r"\b(prices?|priced|pricing|closes?|closed|closing|completes?|completed|announces? (the )?pricing)\b", re.I)
SEVERE = {"ATM Offering", "Direct Offering", "Public/Stock Offering", "Dilution", "Private Placement"}

# ---- أحداث إيجابية ----
POSITIVE = [
    (r"\bearnings\b|\bfinancial results\b|\bquarter(ly)? results\b|\bq[1-4] (20\d\d )?results\b", "نتائج مالية"),
    (r"\bFDA\b|\bPDUFA\b|\bapproval\b|\bclinical trial\b|\bphase [123i]+\b|\btopline\b", "FDA / تجارب سريرية"),
    (r"\bcontract\b|\bawarded?\b|\bpurchase order\b", "عقد"),
    (r"\bpartnership\b|\bcollaboration\b|\bagreement\b", "شراكة / اتفاقية"),
    (r"\bmerger\b|\bacquisition\b|\bacquires?\b|\bbuyout\b|\bstrategic alternatives\b", "اندماج / استحواذ"),
    (r"\bproduct launch\b|\blaunch(es|ing)?\b|\bunveils?\b", "إطلاق منتج"),
    (r"\bannual (general )?meeting\b|\bspecial meeting\b|\bshareholder (vote|meeting)\b|\bstockholder", "اجتماع/تصويت مساهمين"),
    (r"\bregulatory (decision|ruling)\b|\bruling\b|\bhearing\b", "قرار تنظيمي"),
    (r"\binvestor (day|conference|event)\b|\bwebcast\b|\bconference call\b", "فعالية مستثمرين"),
]
FUTURE_HINT = re.compile(r"\b(to (host|report|announce|present|hold|release|participate)|will|scheduled|set to|upcoming|expected|ahead of|plans? to)\b", re.I)

WARN_DAYS = 14        # نافذة البحث عن أخبار التخفيف
CATALYST_DAYS = 30    # نافذة عرض المحفزات
TEMP_EXCLUDE_DAYS = 3 # استبعاد مؤقت إذا كان حدث التخفيف حديثًا جدًا


def _parse_item(it):
    """يدعم شكل yfinance القديم والجديد. يعيد dict أو None."""
    try:
        c = it.get("content") if isinstance(it.get("content"), dict) else None
        if c:
            title = c.get("title")
            date = (c.get("pubDate") or c.get("displayTime") or "")[:10]
            url = ((c.get("canonicalUrl") or {}).get("url")
                   or (c.get("clickThroughUrl") or {}).get("url") or "")
            source = (c.get("provider") or {}).get("displayName") or "Yahoo Finance"
        else:
            title = it.get("title")
            ts = it.get("providerPublishTime")
            date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d") if ts else (it.get("date") or "")
            url = it.get("link") or it.get("url") or ""
            source = it.get("publisher") or it.get("source") or "Yahoo Finance"
        if not title or not date:
            return None
        return {"title": str(title), "date": str(date)[:10], "url": url, "source": source}
    except Exception:
        return None


def _days_ago(date_str, now):
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (now - dt).days
    except Exception:
        return None


def analyze_news(items, earnings_dates, now=None):
    """
    items: [{title,date,url,source}] — عناوين Yahoo
    earnings_dates: قائمة date مستقبلية من Yahoo calendar
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    warnings, catalysts = [], []
    temp_excluded = False
    seen = set()

    for it in items or []:
        title, date = it["title"], it["date"]
        age = _days_ago(date, now)
        if age is None or title in seen:
            continue
        seen.add(title)

        # ---- سلبي ----
        if age <= WARN_DAYS:
            for pat, label in NEGATIVE:
                if re.search(pat, title, re.I):
                    already = bool(ALREADY.search(title))
                    importance = "عالية" if label in SEVERE else "متوسطة"
                    warnings.append({
                        "type": label, "title": title, "date": date, "url": it.get("url", ""),
                        "source": it.get("source", "Yahoo Finance"),
                        "state": "حدث بالفعل" if already else "معلن/محتمل",
                        "importance": importance, "age_days": age,
                    })
                    if label in SEVERE and age <= TEMP_EXCLUDE_DAYS:
                        temp_excluded = True
                    break
            else:
                pass

        # ---- إيجابي (من العنوان: فقط إن دلّ على حدث قادم) ----
        if age <= CATALYST_DAYS and FUTURE_HINT.search(title) and not any(
            re.search(p, title, re.I) for p, _ in NEGATIVE
        ):
            for pat, label in POSITIVE:
                if re.search(pat, title, re.I):
                    catalysts.append({
                        "type": label, "title": title, "date": date, "url": it.get("url", ""),
                        "source": it.get("source", "Yahoo Finance"),
                        "kind": "headline", "days_until": None, "importance": "متوسطة",
                    })
                    break

    # ---- مواعيد النتائج المالية (مستقبلية مؤكدة من Yahoo) ----
    today = now.date()
    for dt in earnings_dates or []:
        try:
            d0 = dt.date() if hasattr(dt, "date") and callable(dt.date) else dt
            days = (d0 - today).days
        except Exception:
            continue
        if 0 <= days <= 60:
            catalysts.append({
                "type": "نتائج مالية", "title": "موعد إعلان النتائج المالية", "date": str(d0),
                "url": "", "source": "Yahoo Finance (calendar)",
                "kind": "calendar", "days_until": days, "importance": "عالية" if days <= 14 else "متوسطة",
            })

    warnings.sort(key=lambda x: x["age_days"])
    catalysts.sort(key=lambda x: (x["days_until"] is None, x["days_until"] if x["days_until"] is not None else 0))
    return {"warnings": warnings, "catalysts": catalysts, "temp_excluded": temp_excluded}


def fetch_news(ticker, now=None):
    """يجلب أخبار وتقويم Yahoo لسهم واحد. لا يرفع استثناء أبدًا."""
    empty = {"warnings": [], "catalysts": [], "temp_excluded": False}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        items = []
        try:
            for it in (t.news or []):
                p = _parse_item(it)
                if p:
                    items.append(p)
        except Exception as e:
            print("news error", ticker, e)
        earnings = []
        try:
            cal = t.calendar
            if isinstance(cal, dict):
                e = cal.get("Earnings Date") or []
                earnings = list(e) if isinstance(e, (list, tuple)) else [e]
        except Exception as e:
            print("calendar error", ticker, e)
        return analyze_news(items, earnings, now=now)
    except Exception as e:
        print("news fetch error", ticker, e)
        return empty
