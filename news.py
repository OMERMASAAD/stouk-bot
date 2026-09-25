# -*- coding: utf-8 -*-
"""
طبقة الأخبار والأحداث — مصدرها Yahoo Finance:
  - عناوين الأخبار: yfinance Ticker.news
  - مواعيد النتائج المالية (مستقبلية): yfinance Ticker.calendar

ما يهم الاستراتيجية:
  🚨 تحذيرات (آخر 5 أيام): طرح عام/طرح مباشر/حقوق أولوية (تخفيف)، تقسيم عكسي،
     تحقيق SEC، إنذار شطب، إفلاس، أو نتائج/إيرادات مخيبة.
  📅 محفزات مستقبلية (خلال 7–30 يومًا): موعد النتائج المالية، FDA/PDUFA/تجارب،
     تصويت مساهمين، اندماج، إطلاق منتج... (تُستخرج من العناوين التي تدل على حدث قادم).

حدود Yahoo: لا توجد فيه وثائق SEC ولا صفحات علاقات المستثمرين، والتاريخ المستقبلي
المؤكد المتاح هو موعد النتائج المالية؛ وبقية المحفزات تُستخرج من العناوين.
"""
import re
from datetime import datetime, timezone

# ---- أحداث سلبية (تخفيف/أزمات) ----
NEGATIVE = [
    (r"\breverse (stock )?split\b|\bsplit adjustment\b|\b1-for-\d+\b|\b\d+-for-1\b", "تقسيم عكسي (Reverse Split)"),
    (r"\brights? (issue|offering)\b|\bsubscription rights\b", "إصدار حقوق أولوية (Rights Issue)"),
    (r"\bat[- ]the[- ]market\b|\bATM (offering|program|facility)\b", "طرح ATM (تخفيف)"),
    (r"\bdirect (registered )?offering\b|\bregistered direct\b", "طرح مباشر (Direct Offering)"),
    (r"\bpublic offering\b|\bunderwritten offering\b|\bstock offering\b|\bshare offering\b|\boffering of\b|\bproposed offering\b|\bpricing of\b.*\boffering\b", "طرح عام للأسهم (Public Offering)"),
    (r"\bprivate placement\b|\bPIPE\b", "طرح خاص (Private Placement)"),
    (r"\bdilut\w*", "تخفيف (Dilution)"),
    (r"\bwarrants?\b", "إصدار Warrants"),
    (r"\bconvertible (senior )?notes?\b|\bconvertible debenture", "سندات قابلة للتحويل"),
    (r"\bregistration statement\b|\bshelf (registration|offering)\b|\bform S-[13]\b|\bS-3\b", "تسجيل أسهم جديدة (Shelf)"),
    (r"\bsec (investigation|subpoena|probe|inquiry)\b|\binvestigation by the (sec|securities)\b|\bsecurities investigation\b", "تحقيق SEC"),
    (r"\bdelist\w*|\bnasdaq (notice|non-?compliance|deficiency)\b|\bminimum (bid|stockholders. equity) requirement\b", "إنذار شطب (Delisting)"),
    (r"\bbankrupt\w*|\bchapter 11\b|\bchapter 7\b|\bgoing concern\b|\binsolven\w*|\breceivership\b", "إفلاس/مخاوف استمرارية"),
    (r"\bmisses? (revenue|earnings|estimates)\b|\bmissed estimates\b|\brevenue (decline|crashes|falls|drop)\b|\bguides? (down|below)\b|\blowers? (outlook|guidance)\b|\bcuts? (outlook|guidance|forecast)\b|\bimpairment\b|\bnet loss widens?\b", "نتائج مخيبة/إيرادات متراجعة"),
]
# كلمات تدل أن الحدث تم فعلًا
ALREADY = re.compile(r"\b(prices?|priced|pricing|closes?|closed|closing|completes?|completed|announces? (the )?pricing)\b", re.I)
SEVERE = {
    "طرح ATM (تخفيف)", "طرح مباشر (Direct Offering)", "طرح عام للأسهم (Public Offering)",
    "تخفيف (Dilution)", "طرح خاص (Private Placement)", "إصدار حقوق أولوية (Rights Issue)",
    "تسجيل أسهم جديدة (Shelf)", "إفلاس/مخاوف استمرارية",
}

# ---- أحداث إيجابية (محفزات) ----
POSITIVE = [
    (r"\bearnings\b|\bfinancial results\b|\bquarter(ly)? results\b|\bq[1-4] (20\d\d )?results\b|\bfiscal (year|quarter)\b", "📅 نتائج مالية"),
    (r"\bFDA\b|\bPDUFA\b|\bapproval\b|\bclinical trial\b|\bphase [123i]+ \b|\bphase [123i]+\b|\btopline\b|\bdata readout\b|\bIND\b|\bBLA\b|\bNDA\b", "📅 FDA / تجارب سريرية"),
    (r"\bcontract\b|\bawarded?\b|\bpurchase order\b", "📅 عقد جديد"),
    (r"\bmerger\b|\bacquisition\b|\bacquires?\b|\bbuyout\b|\bstrategic alternatives\b|\bmerger vote\b", "📅 اندماج / استحواذ"),
    (r"\bannual (general )?meeting\b|\bspecial meeting\b|\bshareholder (vote|meeting)\b|\bstockholder (vote|meeting)\b|\bproxy\b", "📅 اجتماع/تصويت مساهمين"),
    (r"\bproduct launch\b|\blaunch(es|ing)?\b|\bunveils?\b", "📅 إطلاق منتج"),
    (r"\bregulatory (decision|ruling)\b|\bruling\b|\bhearing\b", "📅 قرار تنظيمي"),
    (r"\binvestor (day|conference|event)\b|\bwebcast\b|\bconference call\b|\bpresentation\b", "📅 فعالية مستثمرين"),
    (r"\bpartnership\b|\bcollaboration\b|\bagreement\b", "📅 شراكة / اتفاقية"),
]
FUTURE_HINT = re.compile(r"\b(to (host|report|announce|present|hold|release|participate)|will|scheduled|set to|upcoming|expected|ahead of|plans? to|by the end of|next (week|month|quarter))\b", re.I)

WARN_DAYS = 5          # نافذة التحذيرات: آخر 5 أيام (حسب المواصفات)
CATALYST_DAYS = 30     # أقصى مدى لعرض المحفزات
CATALYST_MIN_DAYS = 7  # المحفز الفعلي يبدأ من 7 أيام
TEMP_EXCLUDE_DAYS = 3  # استبعاد مؤقت (عرضي) إذا كان حدث التخفيف حديثًا جدًا

AR_MONTHS = {
    1: "يناير", 2: "فبراير", 3: "مارس", 4: "أبريل", 5: "مايو", 6: "يونيو",
    7: "يوليو", 8: "أغسطس", 9: "سبتمبر", 10: "أكتوبر", 11: "نوفمبر", 12: "ديسمبر",
}


def _ar_date(date_str, now=None):
    try:
        dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    except Exception:
        return str(date_str)
    label = f"{dt.day} {AR_MONTHS.get(dt.month, '')}"
    year_now = (now or datetime.now(timezone.utc)).year
    if dt.year != year_now:
        label += f" {dt.year}"
    return label.strip()


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
        dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return (now - dt).days
    except Exception:
        return None


def analyze_news(items, earnings_dates, now=None):
    """
    items: [{title,date,url,source}] — عناوين Yahoo
    earnings_dates: قائمة تواريخ مستقبلية من Yahoo calendar
    يعيد {"warnings", "catalysts", "temp_excluded"}
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

        is_negative = False
        if 0 <= age <= WARN_DAYS:
            for pat, label in NEGATIVE:
                if re.search(pat, title, re.I):
                    already = bool(ALREADY.search(title))
                    warnings.append({
                        "type": label, "title": title, "date": date, "url": it.get("url", ""),
                        "source": it.get("source", "Yahoo Finance"),
                        "state": "حدث بالفعل" if already else "معلن/محتمل",
                        "severity": "عالية" if label in SEVERE else "متوسطة",
                        "importance": "عالية" if label in SEVERE else "متوسطة",
                        "age_days": age,
                    })
                    if label in SEVERE and age <= TEMP_EXCLUDE_DAYS:
                        temp_excluded = True
                    is_negative = True
                    break

        # ---- إيجابي: فقط العنوان الذي يدل على حدث قادم ----
        if not is_negative and age <= CATALYST_DAYS and FUTURE_HINT.search(title):
            for pat, label in POSITIVE:
                if re.search(pat, title, re.I):
                    catalysts.append({
                        "type": label, "title": title, "date": date, "url": it.get("url", ""),
                        "source": it.get("source", "Yahoo Finance"),
                        "kind": "headline", "days_until": None, "importance": "متوسطة",
                    })
                    break

    # ---- مواعيد النتائج المالية (مستقبلية مؤكدة من Yahoo calendar) ----
    today = now.date()
    for dt in earnings_dates or []:
        try:
            d0 = dt.date() if hasattr(dt, "date") and callable(dt.date) else dt
            days = (d0 - today).days
        except Exception:
            continue
        if 0 <= days <= 60:
            catalysts.append({
                "type": "📅 نتائج مالية", "title": "موعد إعلان النتائج المالية",
                "date": str(d0), "url": "", "source": "Yahoo Finance (calendar)",
                "kind": "calendar", "days_until": days,
                "importance": "عالية" if days <= 14 else "متوسطة",
            })

    warnings.sort(key=lambda x: x["age_days"])
    catalysts.sort(key=lambda x: (x["days_until"] is None, x["days_until"] if x["days_until"] is not None else 0))
    return {"warnings": warnings, "catalysts": catalysts, "temp_excluded": temp_excluded}


def summarize_news(news, now=None, min_days=CATALYST_MIN_DAYS, max_days=CATALYST_DAYS):
    """
    يختصر الأخبار إلى ما تحتاجه الواجهة والدرجة:
      has_warning / warning_detail (🚨 تحذير: خافض لقيمة الأسهم / أزمة)
      has_upcoming_catalyst / catalyst_detail (📅 محفز مستقبلي)
    """
    news = news or {}
    warnings = news.get("warnings") or []
    catalysts = news.get("catalysts") or []

    has_warning = bool(warnings)
    warning_detail = ""
    if has_warning:
        top = warnings[0]
        reasons = []
        for w in warnings:
            if w.get("type") not in reasons:
                reasons.append(w.get("type"))
        reason_txt = "، ".join(reasons[:3])
        if len(reasons) > 3:
            reason_txt += f" (+{len(reasons) - 3})"
        warning_detail = (
            f"🚨 تحذير: خافض لقيمة الأسهم / أزمة — {reason_txt} "
            f"({_ar_date(top.get('date'), now)}) · {str(top.get('title', ''))[:150]}"
        )

    upcoming = []
    for c in catalysts:
        days = c.get("days_until")
        if days is None:
            upcoming.append(c)                       # عنوان يدل على حدث قادم
        elif min_days <= days <= max_days:
            upcoming.append(c)
    has_catalyst = bool(upcoming)
    catalyst_detail = ""
    if has_catalyst:
        top = upcoming[0]
        if top.get("kind") == "calendar":
            catalyst_detail = f"📅 موعد نتائج مالية متوقع: {_ar_date(top.get('date'), now)}"
            if top.get("days_until") is not None:
                catalyst_detail += f" (بعد {top['days_until']} يومًا)"
        else:
            catalyst_detail = f"📅 {str(top.get('type', 'محفز')).replace('📅 ', '')}: {str(top.get('title', ''))[:140]}"

    return {
        "has_warning": has_warning,
        "warning_detail": warning_detail,
        "has_upcoming_catalyst": has_catalyst,
        "catalyst_detail": catalyst_detail,
        "temp_excluded": bool(news.get("temp_excluded")),
        "warnings_count": len(warnings),
        "catalysts_count": len(catalysts),
    }


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
