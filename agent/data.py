"""Synthetic payments database for the agent to query ("Northwind Payments", a fictional processor).

Deterministic: the same seed always produces the same rows, so benchmark answers can be computed exactly.
"""
from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "payments.db"

COUNTRIES = ["US", "GB", "DE", "FR", "IN", "SG", "BR", "CA", "AU", "JP"]
CURRENCY = {"US": "USD", "GB": "GBP", "DE": "EUR", "FR": "EUR", "IN": "INR", "SG": "SGD", "BR": "BRL",
            "CA": "CAD", "AU": "AUD", "JP": "JPY"}
# units of currency per 1 USD (static snapshot used by the convert_currency tool)
FX = {"USD": 1.0, "GBP": 0.79, "EUR": 0.92, "INR": 83.2, "SGD": 1.35, "BRL": 5.05, "CAD": 1.36,
      "AUD": 1.52, "JPY": 151.0}
MCC = [("5411", "Grocery stores"), ("5812", "Restaurants"), ("5541", "Fuel stations"), ("5732", "Electronics"),
       ("4722", "Travel agencies"), ("5967", "Direct marketing - inbound telemarketing"), ("7995", "Gambling"),
       ("5999", "Miscellaneous retail"), ("4121", "Taxis and rideshare"), ("5691", "Clothing")]
HIGH_RISK_MCC = {"5967", "7995", "4722"}
REASONS = [("10.4", "Fraud - card-absent environment"), ("13.1", "Merchandise not received"),
           ("13.3", "Not as described"), ("12.6", "Duplicate processing"), ("13.7", "Cancelled merchandise")]
NAMES_A = ["Blue", "Harbor", "Summit", "Maple", "Copper", "Lunar", "Atlas", "Cedar", "Nova", "Pioneer",
           "Crescent", "Granite", "Willow", "Aurora", "Beacon", "Delta", "Evergreen", "Falcon", "Golden", "Ivory"]
NAMES_B = ["Market", "Bistro", "Fuel", "Electronics", "Travel", "Outlet", "Gaming", "Goods", "Rides", "Apparel"]
START = date(2026, 1, 1)
DAYS = 181   # Jan 1 - Jun 30 2026


def build(path: Path = DB_PATH, seed: int = 42) -> Path:
    rng = random.Random(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE merchants (merchant_id TEXT PRIMARY KEY, name TEXT, mcc TEXT, category TEXT, country TEXT,
                            currency TEXT, risk_tier TEXT, onboarded DATE);
    CREATE TABLE transactions (txn_id INTEGER PRIMARY KEY, merchant_id TEXT, txn_date DATE, amount REAL,
                               currency TEXT, status TEXT, card_country TEXT);
    CREATE TABLE chargebacks (chargeback_id INTEGER PRIMARY KEY, txn_id INTEGER, reason_code TEXT,
                              reason TEXT, opened DATE, status TEXT);
    CREATE TABLE fx_rates (currency TEXT PRIMARY KEY, units_per_usd REAL);
    """)
    merchants = []
    for i in range(120):
        mcc, cat = MCC[i % len(MCC)]
        country = rng.choice(COUNTRIES)
        name = f"{NAMES_A[i % 20]} {NAMES_B[i % 10]} {'Co' if i < 60 else 'Ltd'}"
        tier = "high" if mcc in HIGH_RISK_MCC else rng.choice(["low", "low", "medium"])
        onboard = START - timedelta(days=rng.randint(30, 900))
        merchants.append((f"M{1000 + i}", name, mcc, cat, country, CURRENCY[country], tier, onboard.isoformat()))
    cur.executemany("INSERT INTO merchants VALUES (?,?,?,?,?,?,?,?)", merchants)
    cur.executemany("INSERT INTO fx_rates VALUES (?,?)", list(FX.items()))

    txns, cbs = [], []
    tid = 1
    for m in merchants:
        mid, _, mcc, _, country, cur_code, tier, _ = m
        n = rng.randint(150, 600)
        base = {"5411": 40, "5812": 35, "5541": 50, "5732": 220, "4722": 480, "5967": 90, "7995": 120,
                "5999": 60, "4121": 22, "5691": 75}[mcc] * FX[cur_code]
        cb_rate = {"high": 0.018, "medium": 0.006, "low": 0.002}[tier]
        decline = {"high": 0.12, "medium": 0.06, "low": 0.03}[tier]
        for _ in range(n):
            d = START + timedelta(days=rng.randrange(DAYS))
            amt = round(rng.lognormvariate(0, 0.6) * base, 2)
            r = rng.random()
            status = "declined" if r < decline else ("refunded" if r < decline + 0.03 else "approved")
            card_country = country if rng.random() < 0.85 else rng.choice(COUNTRIES)
            txns.append((tid, mid, d.isoformat(), amt, cur_code, status, card_country))
            if status == "approved" and rng.random() < cb_rate:
                code, reason = rng.choice(REASONS)
                opened = d + timedelta(days=rng.randint(5, 60))
                st = rng.choice(["open", "open", "won", "lost"])
                cbs.append((len(cbs) + 1, tid, code, reason, min(opened, START + timedelta(days=DAYS + 30)).isoformat(), st))
            tid += 1
    cur.executemany("INSERT INTO transactions VALUES (?,?,?,?,?,?,?)", txns)
    cur.executemany("INSERT INTO chargebacks VALUES (?,?,?,?,?,?)", cbs)
    cur.executescript("CREATE INDEX ix_t_m ON transactions(merchant_id); CREATE INDEX ix_cb_t ON chargebacks(txn_id);")
    con.commit()
    con.execute("VACUUM")
    con.close()
    return path


SCHEMA_DOC = """Tables (SQLite):
merchants(merchant_id TEXT e.g. 'M1000', name, mcc, category, country ISO-2, currency, risk_tier 'low'|'medium'|'high', onboarded DATE)
transactions(txn_id, merchant_id, txn_date DATE 2026-01-01..2026-06-30, amount REAL in the merchant's currency, currency, status 'approved'|'declined'|'refunded', card_country)
chargebacks(chargeback_id, txn_id, reason_code, reason, opened DATE, status 'open'|'won'|'lost')
fx_rates(currency, units_per_usd)"""

def digest(path: Path) -> str:
    import hashlib
    con = sqlite3.connect(path)
    h = hashlib.sha256("\n".join(con.iterdump()).encode()).hexdigest()
    con.close()
    return h


if __name__ == "__main__":
    import sys
    import tempfile
    if "--check" in sys.argv:  # rebuild from the seed and require identical contents to the committed database
        with tempfile.TemporaryDirectory() as d:
            fresh = digest(build(Path(d) / "payments.db"))
        committed = digest(DB_PATH)
        print("committed", committed[:16], "rebuilt", fresh[:16])
        sys.exit(0 if fresh == committed else "database does not match a fresh build from the seed")
    p = build()
    con = sqlite3.connect(p)
    for t in ("merchants", "transactions", "chargebacks", "fx_rates"):
        print(t, con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0])
    print(f"{p} {p.stat().st_size / 1e6:.1f} MB")
