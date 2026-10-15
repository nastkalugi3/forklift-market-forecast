"""Внешние факторы: сборка квартальных рядов из сырых выгрузок (data/external/raw)
и ручных таблиц с источниками (data/external/manual).

Каждое значение в итоговом long-файле несёт источник, ссылку и дату выгрузки.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_SNAP = Path(__file__).resolve().parents[1] / "data/external/raw/_snapshot.txt"
# дата выгрузки текущего снимка внешних данных (обновляется src/fetch_external.py)
RETRIEVED = _SNAP.read_text(encoding="utf-8").strip() if _SNAP.exists() else "2026-09-25"

SRC = {
    "key_rate": ("Банк России", "https://www.cbr.ru/hd_base/KeyRate/",
                 "Ключевая ставка; среднее по календарным дням квартала (как в опросе ЦБ «с учётом выходных»)"),
    "usd_rub": ("Банк России", "https://www.cbr.ru/scripts/XML_dynamic.asp?VAL_NM_RQ=R01235",
                "Официальный курс USD/RUB; среднее по датам установления курса"),
    "cny_rub": ("Банк России", "https://www.cbr.ru/scripts/XML_dynamic.asp?VAL_NM_RQ=R01375",
                "Официальный курс CNY/RUB (за 1 юань, с учётом номинала); среднее по датам установления курса"),
    "credit": ("Банк России", "https://www.cbr.ru/dataservice/ (publicationId=22, datasetId=49/50, РФ, всего)",
               "Кредиты юрлицам и ИП (вкл. МСП): объём выданных за период и задолженность на конец периода, номинал"),
    "monitoring": ("Банк России", "https://www.cbr.ru/dataservice/ (publicationId=25, datasetId=76/77, сезонно скорр.)",
                   "Мониторинг предприятий: баланс ответов об инвестиционной активности и её ожиданиях, пункты"),
    "bop": ("Банк России", "https://www.cbr.ru/dataservice/ (publicationId=9, datasetId=13)",
            "Платёжный баланс: импорт товаров, млн долл."),
}


def key_rate_quarterly(path: Path, start="2022-01-01", end: str | None = None) -> pd.DataFrame:
    """Средняя ставка по календарным дням квартала. Последнее решение действует до конца квартала,
    в котором сделан снимок данных (RETRIEVED): до следующего заседания ставка известна."""
    if end is None:
        end = str(pd.Period(RETRIEVED, freq="Q").end_time.date())
    ch = pd.read_csv(path, parse_dates=["effective_date"]).sort_values("effective_date")
    days = pd.date_range(start, end, freq="D")
    s = pd.Series(np.nan, index=days)
    for d, v in zip(ch.effective_date, ch.key_rate):
        s[s.index >= d] = v
    q = s.groupby(s.index.to_period("Q"))
    return pd.DataFrame({"key_rate_avg": q.mean().round(3), "key_rate_end": q.last(),
                         "n_days": q.size()})


def credit_quarterly(path: Path) -> pd.DataFrame:
    """Aggregate monthly credit data to quarters; incomplete quarters become NaN."""
    m = pd.read_csv(path)
    m["q"] = pd.PeriodIndex(m.month, freq="M").asfreq("Q")
    g = m.groupby("q")
    out = pd.DataFrame({"credit_new_bln": (g.credit_new_mln.sum() / 1e3).round(1),
                        "credit_debt_end_bln": (g.credit_debt_mln.last() / 1e3).round(1),
                        "n_months": g.size()})
    out.loc[out.n_months < 3, ["credit_new_bln", "credit_debt_end_bln"]] = np.nan  # неполный квартал
    out["credit_new_yoy"] = (out.credit_new_bln.pct_change(4) * 100).round(2)
    out["credit_debt_yoy"] = (out.credit_debt_end_bln.pct_change(4) * 100).round(2)
    return out


def build_macro(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collect raw and manual external series into long and wide quarterly tables."""
    raw, man = root / "data/external/raw", root / "data/external/manual"
    rows = []

    def add(df, col, ind, unit, src_key, note=""):
        org, url, desc = SRC[src_key]
        for q, v in df[col].items():
            rows.append({"quarter": str(q), "indicator": ind, "value": v, "unit": unit,
                         "source_org": org, "publisher": org, "url": url,
                         "retrieved": RETRIEVED, "note": (desc + ("; " + note if note else ""))})

    kr = key_rate_quarterly(raw / "cbr_key_rate_changes.csv")
    add(kr, "key_rate_avg", "key_rate_avg", "% годовых", "key_rate")
    add(kr, "key_rate_end", "key_rate_end", "% годовых", "key_rate")

    fx = pd.read_csv(raw / "cbr_usdrub_quarterly.csv").set_index("quarter")
    add(fx, "usd_rub_avg", "usd_rub_avg", "руб./долл.", "usd_rub",
        f"последний квартал может быть неполным (данные до {RETRIEVED})")
    add(fx, "usd_rub_end", "usd_rub_end", "руб./долл.", "usd_rub")

    cny_path = man / "cbr_cnyrub_daily.csv"
    if cny_path.exists():
        cd = pd.read_csv(cny_path, parse_dates=["date"])
        g = cd.set_index("date")["cny_rub"].groupby(lambda d: d.to_period("Q"))
        cq = pd.DataFrame({"cny_rub_avg": g.mean().round(4), "cny_rub_end": g.last(), "n": g.size()})
        cq = cq[cq.n >= 10]  # квартал с единичными датами не берём
        cq.index = cq.index.astype(str)
        start = len(rows)
        add(cq, "cny_rub_avg", "cny_rub_avg", "руб./юань", "cny_rub",
            f"последний квартал может быть неполным (данные до {cd.date.max().date()})")
        for r in rows[start:]:
            r["retrieved"] = str(cd.retrieved.iloc[-1])

    cr = credit_quarterly(raw / "cbr_credit_legal_entities_monthly.csv")
    for c, u in [("credit_new_bln", "млрд руб."), ("credit_new_yoy", "% г/г, номинал"),
                 ("credit_debt_end_bln", "млрд руб."), ("credit_debt_yoy", "% г/г, номинал")]:
        add(cr, c, c, u, "credit")

    mon = pd.read_csv(raw / "cbr_monitoring_invest.csv").set_index("quarter")
    for c in [c for c in mon if c.startswith("inv_")]:
        add(mon, c, c, "пункты (баланс ответов, SA)", "monitoring")

    bop = pd.read_csv(raw / "cbr_bop_goods_quarterly.csv").set_index("quarter")
    bop["goods_imports_yoy"] = (bop.goods_imports_musd.pct_change(4) * 100).round(2)
    add(bop, "goods_imports_musd", "goods_imports_musd", "млн долл.", "bop")
    add(bop, "goods_imports_yoy", "goods_imports_yoy", "% г/г, в долл.", "bop")

    rs = pd.read_csv(man / "rosstat_via_media.csv")
    for _, r in rs.iterrows():
        rows.append({"quarter": r.quarter, "indicator": r.indicator, "value": r.value, "unit": r.unit,
                     "source_org": r.source_org, "publisher": r.publisher, "url": r.url,
                     "retrieved": RETRIEVED, "note": f"значение на {r.vintage_date}; " + str(r.note if pd.notna(r.note) else "")})

    long = pd.DataFrame(rows)
    long = long[long.quarter >= "2021Q1"].sort_values(["indicator", "quarter"]).reset_index(drop=True)
    wide = long.pivot_table(index="quarter", columns="indicator", values="value", aggfunc="first")
    wide.index = pd.PeriodIndex(wide.index, freq="Q")
    wide = wide[wide.index >= pd.Period("2022Q1", freq="Q")]
    return long, wide
