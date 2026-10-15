"""Траектории внешних факторов для прогноза: факт + сценарные допущения + автопродление.

Используется этапами 04–06, чтобы при добавлении нового квартала прогноз не ломался из-за того,
что сценарные таблицы заканчиваются раньше нового горизонта. Все автоматические достройки
возвращаются списком notes и печатаются в ноутбуке и в сводке обновления.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SCENARIOS = ["Консервативный", "Базовый", "Оптимистичный"]


def load_scenarios(root: Path) -> tuple[pd.DataFrame, dict]:
    """Read scenario paths and the rate pass-through factors from data/external/manual/scenarios.csv."""
    sc_all = pd.read_csv(Path(root) / "data/external/manual/scenarios.csv")
    pas = sc_all[sc_all.factor == "rate_passthrough"].set_index("scenario").value.astype(float).to_dict()
    sc = sc_all[sc_all.factor != "rate_passthrough"].copy()
    sc["quarter"] = pd.PeriodIndex(sc.quarter, freq="Q")
    return sc, pas


def _path(fact: pd.Series, scen: pd.Series, need_to: pd.Period, label: str, notes: list) -> pd.Series:
    s = pd.concat([fact.dropna(), scen])
    s = s[~s.index.duplicated(keep="first")].sort_index()          # факт имеет приоритет над допущением
    if s.index.max() < need_to:
        last_q, last_v = s.index.max(), s.iloc[-1]
        ext = pd.Series(last_v, index=pd.period_range(last_q + 1, need_to, freq="Q"))
        rng = str(need_to) if ext.index.min() == need_to else f"{ext.index.min()}–{need_to}"
        notes.append(f"{label}: нет значений для {rng}; продлено последним значением "
                     f"{last_v:g} ({last_q}). Добавьте строки в data/external/manual/scenarios.csv")
        s = pd.concat([s, ext])
    return s


def scenario_paths(W: pd.DataFrame, root: Path, last: pd.Period, H: int = 8, lag: int = 2):
    """Для каждого сценария: ставка до last+H−lag и инвестиции до last+H (факт → сценарий → продление)."""
    sc, pas = load_scenarios(root)
    notes: list[str] = []
    paths = {}
    for s in SCENARIOS:
        d = sc[sc.scenario == s]
        kr = _path(W["key_rate_avg"], d[d.factor == "key_rate_avg"].set_index("quarter").value,
                   last + H - lag, f"{s}: ключевая ставка", notes)
        inv = _path(W["invest_fc_yoy"], d[d.factor == "invest_fc_yoy"].set_index("quarter").value,
                    last + H, f"{s}: инвестиции", notes)
        paths[s] = {"kr": kr, "inv": inv}
    return paths, pas, notes


def kr_for_forecast(W: pd.DataFrame, root: Path, last: pd.Period, H: int = 4, lag: int = 2):
    """Ключевая ставка для прогноза этапа 04: факт, а где факта нет — базовый сценарий."""
    paths, _, notes = scenario_paths(W, root, last, H=max(H, lag), lag=lag)
    kr = paths["Базовый"]["kr"]
    fact = W["key_rate_avg"].dropna()
    used = [q for q in pd.period_range(last + 1 - lag, last + H - lag, freq="Q") if q not in fact.index]
    if used:
        notes.insert(0, "Ставка для прогноза на 4 кв.: факта нет для " + ", ".join(map(str, used)) +
                     " — взята траектория базового сценария")
    return kr, [n for n in notes if n.startswith("Ставка") or n.startswith("Базовый")]


# ---------------------------------------------------------------- реальная ставка (проверка устойчивости)
def cpi_quarterly(root: Path) -> pd.Series:
    """ИПЦ, % г/г: среднее за квартал по месячным данным Банка России (факт) + прогнозная траектория
    (cpi_forecast.csv: прогноз ЦБ на конец 2026 и 2027 гг., между ними — линейная интерполяция).
    Для незавершённого квартала берётся среднее по доступным месяцам."""
    root = Path(root)
    m = pd.read_csv(root / "data/external/manual/cbr_cpi_monthly.csv")
    q = m.assign(q=pd.PeriodIndex(m.month, freq="M").asfreq("Q")).groupby("q").cpi_yoy.mean()
    f = pd.read_csv(root / "data/external/manual/cpi_forecast.csv")
    f = pd.Series(f.cpi_yoy.to_numpy(), index=pd.PeriodIndex(f.quarter, freq="Q"))
    s = pd.concat([q, f[~f.index.isin(q.index)]]).sort_index()
    # после горизонта прогноза ЦБ инфляция держится на цели 4,0 % (нужно, когда новые кварталы сдвигают горизонт)
    tail = pd.period_range(s.index.max() + 1, s.index.max() + 40, freq="Q")
    return pd.concat([s, pd.Series(4.0, index=tail)])


def real_rate(kr: pd.Series, root: Path) -> pd.Series:
    """Реальная ставка ex post: номинальная ключевая ставка минус ИПЦ г/г того же квартала."""
    cpi = cpi_quarterly(root)
    return (kr - cpi.reindex(kr.index)).dropna()
