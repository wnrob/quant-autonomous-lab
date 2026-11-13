"""Business-day calendar and last-trade (expiry) rules for CME-group futures.

The holiday list is an APPROXIMATION of the CME Globex trading-holiday calendar
(US exchange holidays incl. Good Friday and Juneteenth from 2022). CME publishes the
authoritative calendar per year and occasionally deviates (special closures, e.g.
national days of mourning). Rules below are the published last-trade rules as known
to the agent; verify against the exchange before trading.
"""
from __future__ import annotations
import datetime as dt
from functools import lru_cache

MONTH_CODES = {1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M", 7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z"}
CODE_MONTH = {v: k for k, v in MONTH_CODES.items()}


def _easter(y: int) -> dt.date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a = y % 19; b, c = divmod(y, 100); d, e = divmod(b, 4); f = (b + 8) // 25; g = (b - f + 1) // 3
    hh = (19 * a + b - d - g + 15) % 30; i, k = divmod(c, 4); l = (32 + 2 * e + 2 * i - hh - k) % 7
    m = (a + 11 * hh + 22 * l) // 451; month = (hh + l - 7 * m + 114) // 31; day = (hh + l - 7 * m + 114) % 31 + 1
    return dt.date(y, month, day)


def _nth_weekday(y: int, m: int, weekday: int, n: int) -> dt.date:
    """n-th (1-based) weekday (Mon=0) of month; n=-1 = last."""
    if n > 0:
        d = dt.date(y, m, 1); d += dt.timedelta(days=(weekday - d.weekday()) % 7)
        return d + dt.timedelta(weeks=n - 1)
    nxt = dt.date(y + (m == 12), m % 12 + 1, 1); d = nxt - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: dt.date) -> dt.date:
    if d.weekday() == 5: return d - dt.timedelta(days=1)
    if d.weekday() == 6: return d + dt.timedelta(days=1)
    return d


@lru_cache(maxsize=None)
def holidays(year: int) -> frozenset[dt.date]:
    hs = {_observed(dt.date(year, 1, 1)), _nth_weekday(year, 1, 0, 3), _nth_weekday(year, 2, 0, 3),
          _easter(year) - dt.timedelta(days=2), _nth_weekday(year, 5, 0, -1), _observed(dt.date(year, 7, 4)),
          _nth_weekday(year, 9, 0, 1), _nth_weekday(year, 11, 3, 4), _observed(dt.date(year, 12, 25))}
    if year >= 2022: hs.add(_observed(dt.date(year, 6, 19)))
    # year filter: a Saturday New Year's Day is NOT observed on the prior Dec 31 by US exchanges.
    return frozenset(h for h in hs if h.year == year)


def is_business_day(d: dt.date) -> bool:
    return d.weekday() < 5 and d not in holidays(d.year)


def add_business_days(d: dt.date, n: int) -> dt.date:
    """Move n business days (n<0 = backwards). n=0 returns d unchanged even if d is a holiday."""
    step = 1 if n >= 0 else -1; k = abs(n)
    while k:
        d += dt.timedelta(days=step)
        if is_business_day(d): k -= 1
    return d


def prev_or_same_business_day(d: dt.date) -> dt.date:
    while not is_business_day(d): d -= dt.timedelta(days=1)
    return d


def last_business_day(y: int, m: int) -> dt.date:
    nxt = dt.date(y + (m == 12), m % 12 + 1, 1)
    return prev_or_same_business_day(nxt - dt.timedelta(days=1))


def business_days_between(a: dt.date, b: dt.date) -> int:
    """Number of business days in (a, b]; negative if b < a."""
    if b < a: return -business_days_between(b, a)
    n, d = 0, a
    while d < b:
        d += dt.timedelta(days=1)
        if is_business_day(d): n += 1
    return n


def _prev_month(y: int, m: int) -> tuple[int, int]:
    return (y - 1, 12) if m == 1 else (y, m - 1)


# ---- last-trade rules (contract year y, contract month m) ---------------------------------------------
def lt_equity_index(y: int, m: int) -> dt.date:
    """CME equity index (ES/MES/NQ/MNQ/RTY/M2K): 3rd Friday of contract month (prior business day if holiday)."""
    return prev_or_same_business_day(_nth_weekday(y, m, 4, 3))


def lt_fx(y: int, m: int) -> dt.date:
    """CME FX (6E/6J/6B/6A...): 2 business days before the 3rd Wednesday of contract month."""
    return add_business_days(_nth_weekday(y, m, 2, 3), -2)


def lt_crude(y: int, m: int) -> dt.date:
    """NYMEX CL: 3 business days before the 25th calendar day of the month PRIOR to the contract month;
    if the 25th is not a business day, 3 business days before the business day preceding the 25th."""
    py, pm = _prev_month(y, m)
    return add_business_days(prev_or_same_business_day(dt.date(py, pm, 25)), -3)


def lt_natgas(y: int, m: int) -> dt.date:
    """NYMEX NG: 3 business days prior to the first calendar day of the contract month."""
    return add_business_days(dt.date(y, m, 1), -3)


def lt_comex_metal(y: int, m: int) -> dt.date:
    """COMEX GC/MGC/SI/HG: third-last business day of the contract month."""
    return add_business_days(last_business_day(y, m), -2)


def lt_treasury(y: int, m: int) -> dt.date:
    """CBOT ZN/ZB (and ZF/ZT differ - not covered): 7th business day preceding the last business day of the contract month."""
    return add_business_days(last_business_day(y, m), -7)


def lt_grain(y: int, m: int) -> dt.date:
    """CBOT ZC/ZW/ZS: business day prior to the 15th calendar day of the contract month."""
    return add_business_days(dt.date(y, m, 15), -1)


def fnd_month_before(y: int, m: int) -> dt.date:
    """First notice day for physically delivered COMEX/CBOT contracts: last business day of the month before the contract month
    (CBOT treasuries/grains and COMEX metals). Long holders must exit before this; it is what the market rolls ahead of."""
    py, pm = _prev_month(y, m)
    return last_business_day(py, pm)


EXPIRY_RULES = {"equity_index": lt_equity_index, "fx": lt_fx, "crude": lt_crude, "natgas": lt_natgas,
                "comex_metal": lt_comex_metal, "treasury": lt_treasury, "grain": lt_grain}
