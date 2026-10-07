# -*- coding: utf-8 -*-
"""
المسح الصباحي الشامل (Master Scan): يبني master_low_float.json = رموز منخفضة الفلوت
(Float < 5M) بسعر $0.50–$10 ليقرأها المسح اللحظي كل 15 دقيقة (panic_scanner.py).
يعمل تلقائيًا من scanner.py مرة واحدة في اليوم (أو FORCE_MASTER=1).
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo

import float_lookup

MASTER_FILE = "master_low_float.json"
MASTER_MIN_PRICE, MASTER_MAX_PRICE = 0.50, 10.00
MASTER_MAX_FLOAT = 5_000_000
MIN_AVG_VOLUME = 100_000        # سيولة دنيا (متوسط 20 يومًا) حتى لا نطارد أسهمًا ميتة
POOL_MAX = 700                  # أقصى عدد رموز نبحث لها عن Float في التشغيل الواحد (الكاش يكمل الباقي غدًا)
UNVERIFIED_MAX = 120            # أقصى عدد رموز «Float غير مؤكد» تدخل الماستر (الأعلى تذبذبًا أولًا)
TIME_BUDGET_S = 600


def pool_record(d, price):
    """سجل خفيف لكل سهم داخل نطاق الماستر، من الشموع اليومية التي حُمّلت أصلًا."""
    if not MASTER_MIN_PRICE <= price <= MASTER_MAX_PRICE:
        return None
    avg_vol = float(d["Volume"].tail(20).astype(float).mean())
    if avg_vol < MIN_AVG_VOLUME:
        return None
    rng = ((d["High"] - d["Low"]) / d["Close"]).tail(20).astype(float).mean() * 100
    return {"price": round(price, 3), "avg_vol": int(avg_vol), "range_pct": round(float(rng), 1)}


def et_date(now):
    return now.astimezone(ZoneInfo("America/New_York")).date().isoformat()


def _load():
    try:
        with open(MASTER_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def build_master(pool, now, force=False, resolver=None, workers=8):
    resolver = resolver or float_lookup.resolve
    today, old = et_date(now), _load()
    if not force and old.get("built_on") == today and old.get("tickers"):
        print("master already built today:", len(old["tickers"]))
        return old
    ranked = sorted(pool, key=lambda t: -pool[t]["range_pct"])[:POOL_MAX]
    float_lookup.load_cache()
    deadline, infos = time.time() + TIME_BUDGET_S, {}

    def one(t):
        return None if time.time() > deadline else resolver(t)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, t): t for t in ranked}
        for f in as_completed(futs):
            try:
                infos[futs[f]] = f.result()
            except Exception as e:
                print("master float error", futs[f], type(e).__name__)
    float_lookup.save_cache()

    confirmed, unverified = [], []
    for t in ranked:
        info = infos.get(t)
        if info is None:
            continue                                     # انتهت الميزانية الزمنية — يكمل غدًا من الكاش
        ok, st = float_lookup.classify(info, MASTER_MAX_FLOAT, "watch")
        if not ok:
            continue
        row = dict(pool[t], ticker=t, float=info.get("value"), float_status=st,
                   float_source=info.get("source"))
        (confirmed if st in ("exact", "bound") else unverified).append(row)
    tickers = confirmed + unverified[:UNVERIFIED_MAX]
    if len(tickers) < 20 and old.get("tickers"):
        print("::warning title=Master small::قائمة الماستر الجديدة صغيرة جدًا — أُبقيت القائمة السابقة")
        return old
    out = {"built_on": today, "built_at": now.isoformat(), "count": len(tickers),
           "confirmed": len(confirmed), "unverified": min(len(unverified), UNVERIFIED_MAX),
           "rules": {"price": [MASTER_MIN_PRICE, MASTER_MAX_PRICE], "max_float": MASTER_MAX_FLOAT,
                     "min_avg_volume": MIN_AVG_VOLUME},
           "tickers": tickers}
    with open(MASTER_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("master built:", len(tickers), "(confirmed %d, unverified %d)" % (len(confirmed), min(len(unverified), UNVERIFIED_MAX)))
    return out
