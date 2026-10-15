"""Быстрые автотесты ядра пайплайна (≈30 с): python -m pytest -q tests

Проверяют не «точность прогноза», а инварианты, которые не должны ломаться при правках кода
и при добавлении нового квартала: чтение файла, проверки качества, иерархия, воспроизводимость.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from silant_data import check_formula_ranges, load_market  # noqa: E402
from silant_forecast import MODELS, forecast_all  # noqa: E402
from silant_pipeline import validate_market  # noqa: E402
from silant_ts import build_series  # noqa: E402

XLSX = ROOT / "data/raw/silant_forklift_market.xlsx"


@pytest.fixture(scope="module")
def market():
    long, totals, formulas = load_market(XLSX)
    return long, totals, formulas


def test_file_structure(market):
    long, _, _ = market
    assert long.groupby("type").segment.nunique().to_dict() == {"ДВС": 8, "Электро": 6}
    assert long.quarter.nunique() >= 13
    assert long.units.notna().all() and (long.units >= 0).all()


def test_known_totals(market):
    long, _, _ = market
    y = long.groupby("year").units.sum()
    assert y[2023] == 118_200 and y[2024] == 114_550 and y[2025] == 82_480


def test_formula_range_header_rows_are_harmless(market):
    long, _, formulas = market
    fr = check_formula_ranges(long, formulas)
    # «Итого» электро захватывает название блока и шапку — это не ошибка
    assert (fr.exact_match | fr.header_only_extra).all()


def test_series_hierarchy(market):
    S = build_series(market[0])
    dvs = [c for c in S.columns if c.startswith("ДВС ")]
    el = [c for c in S.columns if c.startswith("Электро ")]
    assert np.allclose(S[dvs].sum(axis=1), S["ДВС"])
    assert np.allclose(S[el].sum(axis=1), S["Электро"])
    assert np.allclose(S["ДВС"] + S["Электро"], S["Рынок"])


def _broken_copy(tmp_path, value):
    p = tmp_path / "broken.xlsx"
    shutil.copy(XLSX, p)
    wb = openpyxl.load_workbook(p)
    wb.worksheets[0].cell(3, 3).value = value   # ДВС 1–1,8 т, 1 кв. 2023
    wb.save(p)
    return p


@pytest.mark.parametrize("value", [-10, None])
def test_validation_stops_on_bad_values(tmp_path, value):
    res = validate_market(_broken_copy(tmp_path, value))
    assert res["errors"], "отрицательное или пустое значение должно останавливать обновление"


def test_validation_passes_on_customer_file():
    res = validate_market(XLSX)
    assert not res["errors"]


@pytest.fixture(scope="module")
def small_forecast(market):
    S = build_series(market[0])
    kr = pd.Series(16.0, index=pd.period_range("2022Q1", S.index.max() + 4, freq="Q"))
    return forecast_all(S, kr, S.index.max(), H=4, n_sim=300, seed=42, infl={m: 1.5 for m in MODELS})


def test_forecast_reconciled_every_path(small_forecast):
    sims = small_forecast["rec"]["Комбинация"]
    for t in ["ДВС", "Электро"]:
        parts = sum(sims[b] for b in small_forecast["bottom"] if b.startswith(t + " "))
        assert np.allclose(parts, sims[t])
    assert np.allclose(sims["ДВС"] + sims["Электро"], sims["Рынок"])
    assert (sims["Рынок"] >= 0).all()


def test_forecast_is_deterministic(market, small_forecast):
    S = build_series(market[0])
    kr = pd.Series(16.0, index=pd.period_range("2022Q1", S.index.max() + 4, freq="Q"))
    again = forecast_all(S, kr, S.index.max(), H=4, n_sim=300, seed=42, infl={m: 1.5 for m in MODELS})
    assert np.array_equal(again["rec"]["Комбинация"]["Рынок"], small_forecast["rec"]["Комбинация"]["Рынок"])


def test_plan_level_rule():
    """Правило уровня плана: α* = Cu/(Cu+Co); множители квантилей монотонны, при α=0,5 равны 1."""
    from silant_forecast import critical_ratio, plan_multipliers

    assert critical_ratio(1, 1) == 0.5
    assert abs(critical_ratio(1, 2) - 1 / 3) < 1e-12
    with pytest.raises(ValueError):
        critical_ratio(0, 0)
    rng = np.random.default_rng(0)
    m = plan_multipliers({"x": rng.lognormal(10, 0.3, 5000)}, [0.1, 0.25, 0.5, 0.75, 0.9])
    assert np.all(np.diff(m.loc["x"].to_numpy()) > 0)
    assert abs(m.loc["x", 0.5] - 1) < 1e-12


def test_dvs_rule(market):
    """Решающее правило по ДВС (отчёт, 5.3): ожидание → предварительно по II кв. (граница 7,7 тыс.) → решение по сумме II + III кв."""
    from silant_pipeline import dvs_rule

    long = market[0][["type", "segment", "quarter", "units"]].copy()
    long["quarter"] = long.quarter.astype(str)
    assert dvs_rule(long)["status"] == "ожидание"

    def add(df, q, dvs_total):
        row = pd.DataFrame({"type": ["ДВС", "Электро"], "segment": ["1–1,8 т", "1–1,8 т"], "quarter": [q, q],
                            "units": [dvs_total, 1000]})
        return pd.concat([df, row], ignore_index=True)

    assert dvs_rule(add(long, "2026Q2", 6_500))["verdict"].startswith("предварительно: спад")
    assert dvs_rule(add(long, "2026Q2", 8_000))["verdict"].startswith("предварительно: признак восстановления")
    assert dvs_rule(add(long, "2026Q2", 7_690))["verdict"].startswith("предварительно: спад")   # граница ≈7,7 тыс., как в отчёте и Excel
    two = add(add(long, "2026Q2", 6_500), "2026Q3", 6_500)
    assert dvs_rule(two)["verdict"] == "спад сохраняется"
    two = add(add(long, "2026Q2", 8_500), "2026Q3", 8_500)
    assert dvs_rule(two)["verdict"] == "восстановление"


def test_cpi_extends_past_forecast():
    """ИПЦ для реальной ставки продлён после горизонта прогноза ЦБ (цель 4 %) — нужно, когда добавлено несколько кварталов."""
    from silant_scenarios import cpi_quarterly

    cpi = cpi_quarterly(ROOT)
    assert cpi.index.is_monotonic_increasing and not cpi.index.duplicated().any()
    assert cpi.index.max() >= pd.Period("2030Q4", freq="Q")
    assert cpi[pd.Period("2029Q1", freq="Q")] == 4.0


def test_persist_exog_cuts_series_at_origin():
    """persist_exog keeps values up to the origin and holds the last known one afterwards."""
    from silant_forecast import persist_exog

    idx = pd.period_range("2022Q1", "2027Q4", freq="Q")
    x = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    origin = pd.Period("2024Q4", freq="Q")
    out = persist_exog(x, origin, 4)
    assert out.index.max() == origin + 4
    assert (out.loc[:origin] == x.loc[:origin]).all()
    assert (out.loc[origin + 1:] == x[origin]).all()


def test_backtest_does_not_use_rate_after_origin(market):
    """The rolling-origin backtest must not see the rate after the origin (lag 2, so h >= 3 used to leak).

    Changing the actual rate after the origin must not change backtest forecasts;
    with persist_exogenous=False (old oracle behaviour) it does."""
    from silant_forecast import rolling_origin

    S = build_series(market[0])
    last = S.index.max()
    origin = last - 3        # h = 3 reaches the last observed quarter
    kr = pd.Series(10.0 + np.arange(len(pd.period_range("2022Q1", last + 4, freq="Q"))) * 0.1,
                   index=pd.period_range("2022Q1", last + 4, freq="Q"))
    kr_shock = kr.copy()
    kr_shock.loc[kr_shock.index > origin] += 8.0       # change the rate after the origin
    kw = dict(H=4, n_sim=100, seed=1, reconcile=False)
    a = rolling_origin(S, kr, origin, origin, **kw)
    b = rolling_origin(S, kr_shock, origin, origin, **kw)
    cols = ["origin", "h", "model", "node", "point"]
    pd.testing.assert_frame_equal(a[cols], b[cols])
    # the old behaviour (oracle rate) does depend on the future rate: the test would catch a regression
    a0 = rolling_origin(S, kr, origin, origin, persist_exogenous=False, **kw)
    b0 = rolling_origin(S, kr_shock, origin, origin, persist_exogenous=False, **kw)
    arx = lambda d: d[(d.model == "ARX-ставка") & (d.node == "Рынок")].point.to_numpy()   # noqa: E731
    assert not np.allclose(arx(a0), arx(b0))
