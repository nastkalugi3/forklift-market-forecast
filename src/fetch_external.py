"""Обновление сырых рядов Банка России (data/external/raw) при появлении новых данных.

Запуск: python src/fetch_external.py   (нужен доступ к https://www.cbr.ru)
Ряды Росстата (инвестиции, промпроизводство) и прогнозы Минэка обновляются вручную
в data/external/manual/*.csv — сайты rosstat.gov.ru / economy.gov.ru недоступны из
среды, где собирался снимок; для каждого значения там указаны источник и дата.
"""
from __future__ import annotations

import datetime as dt
import io
import re
from pathlib import Path

import pandas as pd
import requests

RAW = Path(__file__).resolve().parents[1] / "data/external/raw"
UA = {"User-Agent": "Mozilla/5.0 (silant-forecast)"}
TODAY = dt.date.today()


def key_rate():
    """Ключевая ставка: таблица на cbr.ru/hd_base/KeyRate → даты вступления в силу новых значений."""
    url = ("https://www.cbr.ru/hd_base/KeyRate/?UniDbQuery.Posted=True&UniDbQuery.From=01.01.2022"
           f"&UniDbQuery.To={TODAY:%d.%m.%Y}")
    html = requests.get(url, headers=UA, timeout=60).text
    t = pd.read_html(io.StringIO(html), decimal=",", thousands=" ")[0]
    t.columns = ["date", "rate"]
    t["date"] = pd.to_datetime(t["date"], format="%d.%m.%Y")
    t["rate"] = pd.to_numeric(t["rate"].astype(str).str.replace(",", "."), errors="coerce")
    t = t.dropna().sort_values("date")
    # таблица содержит только рабочие дни; новое значение начинает действовать с первой даты, где оно появилось
    ch = t[t.rate.ne(t.rate.shift())]
    ch = ch.assign(effective_date=ch.date.dt.date, key_rate=ch.rate)[["effective_date", "key_rate"]]
    ch.to_csv(RAW / "cbr_key_rate_changes.csv", index=False)


def usd_rub():
    """Fetch the official USD/RUB rate (R01235) from cbr.ru and save quarterly averages."""
    url = ("https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=01/01/2022&date_req2="
           f"{TODAY:%d/%m/%Y}&VAL_NM_RQ=R01235")
    t = requests.get(url, headers=UA, timeout=60).text
    recs = re.findall(r'<Record Date="([\d.]+)"[^>]*>.*?<Value>([\d,]+)</Value>', t, flags=re.S)
    s = pd.Series([float(v.replace(",", ".")) for _, v in recs],
                  index=pd.to_datetime([d for d, _ in recs], format="%d.%m.%Y"))
    g = s.groupby(s.index.to_period("Q"))
    pd.DataFrame({"usd_rub_avg": g.mean().round(3), "n_obs": g.size(), "usd_rub_end": g.last()}) \
        .rename_axis("quarter").to_csv(RAW / "cbr_usdrub_quarterly.csv")


def cny_rub():
    """Официальный курс юаня (R01375) по дням → data/external/manual/cbr_cnyrub_daily.csv."""
    url = ("https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=01/01/2022&date_req2="
           f"{TODAY:%d/%m/%Y}&VAL_NM_RQ=R01375")
    t = requests.get(url, headers=UA, timeout=60).text
    recs = re.findall(r'<Record Date="([\d.]+)"[^>]*>\s*<Nominal>(\d+)</Nominal>\s*<Value>([\d,]+)</Value>', t)
    df = pd.DataFrame([{"date": pd.to_datetime(d, format="%d.%m.%Y").date(), "nominal": int(n),
                        "value": float(v.replace(",", ".")), "cny_rub": round(float(v.replace(",", ".")) / int(n), 4)}
                       for d, n, v in recs])
    df["source"] = "Банк России, официальный курс юаня (R01375)"
    df["url"] = url
    df["retrieved"] = f"{TODAY:%Y-%m-%d}"
    df.to_csv(RAW.parent / "manual/cbr_cnyrub_daily.csv", index=False)


def dataservice(pub, ds, measure=None, y1=2021):
    """Request one dataset from the Bank of Russia dataservice API and return parsed JSON."""
    url = f"https://www.cbr.ru/dataservice/data?y1={y1}&y2={TODAY.year}&publicationId={pub}&datasetId={ds}"
    if measure is not None:
        url += f"&measureId={measure}"
    return requests.get(url, headers=UA, timeout=60).json()


def credit():
    """Download monthly new credit and debt of legal entities from the Bank of Russia."""
    vol, debt = dataservice(22, 49, 22), dataservice(22, 50, 22)
    v = {r["date"][:7]: r["obs_val"] for r in vol["RawData"] if r["element_id"] == 35}
    d = {r["date"][:7]: r["obs_val"] for r in debt["RawData"] if r["element_id"] == 35}
    rows = [{"month": str(pd.Period(k, freq="M") - 1), "api_date": k + "-01",
             "credit_new_mln": v[k], "credit_debt_mln": d.get(k)} for k in sorted(v)]
    pd.DataFrame(rows).to_csv(RAW / "cbr_credit_legal_entities_monthly.csv", index=False)


def _quarter(label):  # «II квартал 2026» -> 2026Q2
    rom = {"I": 1, "II": 2, "III": 3, "IV": 4}
    r, _, y = label.split()
    return f"{y}Q{rom[r]}"


def monitoring():
    """Download the Bank of Russia enterprise-monitoring investment series."""
    want = {72: "econ", 73: "industry", 85: "wholesale", 87: "transport_storage"}
    out = {}
    for ds, pref in [(76, "inv_act_"), (77, "inv_exp_")]:
        for r in dataservice(25, ds, 119)["RawData"]:
            if r["element_id"] in want:
                out.setdefault(r["dt"], {})[pref + want[r["element_id"]]] = r["obs_val"]
    df = pd.DataFrame(out).T
    df.insert(0, "label", df.index)
    df.index = [_quarter(i) for i in df.index]
    df.rename_axis("quarter").sort_index().to_csv(RAW / "cbr_monitoring_invest.csv")


def bop_goods():
    """Download quarterly goods exports, imports and balance from the balance of payments."""
    d = dataservice(9, 13)
    m = {}
    for r in d["RawData"]:
        m.setdefault(r["dt"], {})[r["element_id"]] = r["obs_val"]
    rows = [{"quarter": _quarter(k), "label": k, "goods_balance_musd": v.get(28),
             "goods_exports_musd": v.get(29), "goods_imports_musd": v.get(30)} for k, v in m.items()]
    pd.DataFrame(rows).sort_values("quarter").to_csv(RAW / "cbr_bop_goods_quarterly.csv", index=False)



def cpi():
    """ИПЦ, % г/г по месяцам: таблица «Инфляция и ключевая ставка Банка России» (cbr.ru/hd_base/infl).
    Пишет data/external/manual/cbr_cpi_monthly.csv (тот же формат, что и ручной снимок)."""
    url = ("https://www.cbr.ru/hd_base/infl/?UniDbQuery.Posted=True&UniDbQuery.From=01.01.2021"
           f"&UniDbQuery.To={TODAY:%d.%m.%Y}")
    html = requests.get(url, headers=UA, timeout=60).text
    t = pd.read_html(io.StringIO(html), decimal=",", thousands=" ")[0]
    t = t.iloc[:, [0, 2]]
    t.columns = ["month", "cpi_yoy"]
    t["month"] = pd.PeriodIndex(pd.to_datetime(t["month"].astype(str), format="%m.%Y"), freq="M").astype(str)
    t["cpi_yoy"] = pd.to_numeric(t["cpi_yoy"].astype(str).str.replace(",", "."), errors="coerce")
    t = t.dropna().sort_values("month")
    t["source"] = "Банк России, «Инфляция и ключевая ставка Банка России» (ИПЦ, % г/г, по данным Росстата)"
    t["url"] = "https://www.cbr.ru/hd_base/infl/"
    t["retrieved"] = str(TODAY)
    t.to_csv(RAW.parent / "manual/cbr_cpi_monthly.csv", index=False)

if __name__ == "__main__":
    ok = 0
    for f in (key_rate, usd_rub, cny_rub, credit, monitoring, bop_goods, cpi):
        try:
            f()
            print("OK", f.__name__)
            ok += 1
        except Exception as e:  # noqa: BLE001
            print("FAIL", f.__name__, e)
    if ok:
        (RAW / "_snapshot.txt").write_text(TODAY.isoformat(), encoding="utf-8")
    print("Ряды Росстата и прогнозы (data/external/manual/*.csv) обновляются вручную — см. data/external/SOURCES.md")
