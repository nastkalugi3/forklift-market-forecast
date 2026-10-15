"""Тренд и сезонность на коротких квартальных рядах (13 точек).

Все модели — на логарифмах: коэффициенты читаются как % изменения.
Базовая спецификация: log(y) = a + b·t [+ c·1(t ≥ break)] [+ сезонные дамми 2–4 кв.].
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

BREAK = pd.Period("2025Q2", freq="Q")  # дата слома ДВС из EDA (этап 01)


# ------------------------------------------------------------------ ряды
def build_series(long: pd.DataFrame, small_from: float = 5.5) -> pd.DataFrame:
    """Широкая таблица рядов: крупные сегменты, агрегат малых (≥5,5 т) по типу, типы, рынок."""
    d = long.assign(grp=np.where(long.cap_min >= small_from, f"{small_from:g} т+".replace(".", ","),
                                 long.segment))
    out = {}
    for (t, g), x in d.groupby(["type", "grp"], sort=False):
        out[f"{t} {g}"] = x.groupby("quarter").units.sum()
    for t, x in d.groupby("type"):
        out[t] = x.groupby("quarter").units.sum()
    out["Рынок"] = d.groupby("quarter").units.sum()
    return pd.DataFrame(out)


def design(idx: pd.PeriodIndex, trend=True, step: pd.Period | None = None, seas=True) -> pd.DataFrame:
    """Regression design matrix: constant, linear trend, structural step and quarter dummies."""
    n = len(idx)
    X = {"const": np.ones(n)}
    if trend:
        X["trend"] = np.arange(n, dtype=float)
    if step is not None:
        X["step"] = (idx >= step).astype(float)
    if seas:
        for k in (2, 3, 4):
            X[f"Q{k}"] = (idx.quarter == k).astype(float)
    return pd.DataFrame(X, index=idx)


def aicc(res) -> float:
    """Small-sample corrected AIC of a fitted OLS model."""
    k = res.df_model + 1 + 1  # параметры регрессии + дисперсия
    n = res.nobs
    return res.aic + 2 * k * (k + 1) / (n - k - 1)


def compare_specs(y: pd.Series, step: pd.Period = BREAK) -> pd.DataFrame:
    """AICc 8 спецификаций: {уровень, тренд} × {без слома, слом} × {без сезона, сезон}."""
    rows = []
    ly = np.log(y.clip(lower=1))
    for trend, st, seas in itertools.product([False, True], [None, step], [False, True]):
        r = sm.OLS(ly, design(y.index, trend, st, seas)).fit()
        rows.append({"trend": trend, "break": st is not None, "season": seas,
                     "k": int(r.df_model + 1), "sigma": np.sqrt(r.scale), "AICc": aicc(r)})
    df = pd.DataFrame(rows)
    df["dAICc"] = df.AICc - df.AICc.min()
    return df


def break_scan(y: pd.Series, trend=False, seas=True, first=3, last_offset=2) -> pd.Series:
    """Сумма квадратов остатков модели со ступенькой для каждой возможной даты слома."""
    ly = np.log(y.clip(lower=1))
    out = {}
    for b in y.index[first:len(y) - last_offset]:
        out[b] = sm.OLS(ly, design(y.index, trend, b, seas)).fit().ssr
    return pd.Series(out)


def sup_f_break(y: pd.Series, trend=False, n_boot=2000, seed=42) -> dict:
    """Тест на слом с неизвестной датой (sup-F, Andrews) с параметрическим бутстрепом
    критических значений: дата выбирается по данным, поэтому обычный F-тест завышает значимость."""
    rng = np.random.default_rng(seed)
    ly = np.log(y.clip(lower=1)).to_numpy()
    idx = y.index
    X0 = design(idx, trend, None, True).to_numpy()
    cands = idx[3:len(idx) - 2]
    Xb = [design(idx, trend, b, True).to_numpy() for b in cands]

    def supf(v):
        r0 = np.linalg.lstsq(X0, v, rcond=None)
        ssr0 = ((v - X0 @ r0[0]) ** 2).sum()
        best = 0
        for X in Xb:
            b = np.linalg.lstsq(X, v, rcond=None)[0]
            ssr1 = ((v - X @ b) ** 2).sum()
            best = max(best, (ssr0 - ssr1) / (ssr1 / (len(v) - X.shape[1])))
        return best

    obs = supf(ly)
    b0 = np.linalg.lstsq(X0, ly, rcond=None)[0]
    fit0 = X0 @ b0
    s0 = np.sqrt(((ly - fit0) ** 2).sum() / (len(ly) - X0.shape[1]))
    sims = np.array([supf(fit0 + rng.normal(0, s0, len(ly))) for _ in range(n_boot)])
    return {"supF": obs, "p_boot": float((sims >= obs).mean())}


# ------------------------------------------------------------------ тренд + сезон
def fit_ts(y: pd.Series, trend=True, step: pd.Period | None = BREAK, seas=True, alpha=0.10):
    """Fit log(y) on trend, break and seasonality by OLS; returns the fit and alpha."""
    ly = np.log(y.clip(lower=1))
    return sm.OLS(ly, design(y.index, trend, step, seas)).fit(), alpha


def seasonal_factors(res, alpha=0.10) -> pd.DataFrame:
    """Сезонные коэффициенты (среднее = 1) с доверительным интервалом (дельта-метод на логах)."""
    names = ["Q2", "Q3", "Q4"]
    b = np.r_[0, res.params[names].to_numpy()]
    V = np.zeros((4, 4))
    V[1:, 1:] = res.cov_params().loc[names, names].to_numpy()
    C = np.eye(4) - 1 / 4  # центрирование: s_k - mean(s)
    s = C @ b
    se = np.sqrt(np.diag(C @ V @ C.T))
    t = stats.t.ppf(1 - alpha / 2, res.df_resid)
    return pd.DataFrame({"factor": np.exp(s), "lo": np.exp(s - t * se), "hi": np.exp(s + t * se),
                         "se_log": se}, index=["1 кв", "2 кв", "3 кв", "4 кв"])


def season_tests(res) -> dict:
    """F-тест совместной значимости сезонных дамми и контраст «1 кв против остальных»."""
    names = ["Q2", "Q3", "Q4"]
    R = np.zeros((3, len(res.params)))
    for i, nm in enumerate(names):
        R[i, list(res.params.index).index(nm)] = 1
    f = res.f_test(R)
    c = pd.Series(0.0, index=res.params.index)
    c[names] = -1 / 3  # Q1 (база=0) минус среднее Q2..Q4
    est = float(c @ res.params)
    se = float(np.sqrt(c @ res.cov_params() @ c))
    df = res.df_resid
    mde = (stats.t.ppf(0.95, df) + stats.t.ppf(0.80, df)) * se  # α=0,10 двуст., мощность 80%
    return {"F_p": float(f.pvalue), "q1_effect": np.expm1(est), "q1_se_log": se,
            "q1_p": float(2 * stats.t.sf(abs(est / se), df)), "mde_q1": -np.expm1(-mde),
            "sigma": float(np.sqrt(res.scale)), "df_resid": int(df)}


def mde_by_years(sigma: float, years=(3, 5, 8, 10)) -> pd.Series:
    """Минимально обнаружимый провал 1-го квартала (α=0,10, мощность 80%) при N полных годах,
    приближённо: SE контраста = σ·sqrt(1/N + 1/(3N)), df = 4N − 6."""
    out = {}
    for N in years:
        se = sigma * np.sqrt(1 / N + 1 / (3 * N))
        df = max(4 * N - 6, 1)
        out[N] = -np.expm1(-(stats.t.ppf(0.95, df) + stats.t.ppf(0.80, df)) * se)
    return pd.Series(out, name="MDE")


def perm_test_shares(y: pd.Series, years) -> tuple[np.ndarray, float]:
    """Точный перестановочный тест: перемешиваем метки кварталов внутри каждого года
    (все 24^N комбинаций). Статистика — разброс средних долей кварталов вокруг 25%.
    Непараметрический, но доли внутри года смещены трендом (при росте 1 кв занижен)."""
    M = np.array([[y[pd.Period(f"{yr}Q{k}", freq="Q")] for k in range(1, 5)] for yr in years], float)
    Sh = M / M.sum(1, keepdims=True)
    stat = lambda A: ((A.mean(0) - .25) ** 2).sum()
    obs = stat(Sh)
    perms = np.array(list(itertools.permutations(range(4))))
    P = [Sh[i][perms] for i in range(len(years))]  # (24,4) на год
    grids = np.meshgrid(*[np.arange(24)] * len(years), indexing="ij")
    acc = np.zeros((grids[0].size, 4))
    for i, g in enumerate(grids):
        acc += P[i][g.ravel()]
    vals = ((acc / len(years) - .25) ** 2).sum(1)
    return Sh.mean(0), float((vals >= obs - 1e-15).mean())


def classical_decomp(y: pd.Series) -> pd.DataFrame:
    """Классическая мультипликативная декомпозиция: тренд = центрированная MA 2×4."""
    trend = y.rolling(4).mean().rolling(2).mean().shift(-2)
    ratio = y / trend
    si = ratio.groupby(y.index.quarter).mean()
    si = si / si.mean()
    return pd.DataFrame({"y": y, "trend": trend, "ratio": ratio,
                         "seasonal": y.index.quarter.map(si).to_numpy(),
                         "sa": y / y.index.quarter.map(si).to_numpy()})


def shrink_seasonality(est: pd.DataFrame, se: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Эмпирический Байес (DerSimonian–Laird) по каждому кварталу: сегментные сезонные
    лог-эффекты стягиваются к общему профилю группы пропорционально своей неточности.
    est/se: строки — сегменты, столбцы — кварталы (центрированные лог-эффекты)."""
    shrunk = est.copy()
    pooled, tau = {}, {}
    for q in est.columns:
        b, s2 = est[q].to_numpy(), se[q].to_numpy() ** 2
        w = 1 / s2
        mu_fe = (w * b).sum() / w.sum()
        Q = (w * (b - mu_fe) ** 2).sum()
        t2 = max(0.0, (Q - (len(b) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
        w2 = 1 / (s2 + t2)
        mu = (w2 * b).sum() / w2.sum()
        B = s2 / (s2 + t2) if t2 > 0 else np.ones_like(s2)
        shrunk[q] = B * mu + (1 - B) * b
        pooled[q], tau[q] = mu, np.sqrt(t2)
    return shrunk, pd.Series(pooled), pd.Series(tau)


# ------------------------------------------------------------------ практическая проверка
def rolling_cv(y: pd.Series, first_train=8, h_max=4, season_ref: pd.Series | None = None) -> pd.DataFrame:
    """Прогноз «с фиксированным началом, растущим окном»: ошибки |log(факт/прогноз)| по методам.
    season_ref — ряд, по которому оценивается общий (pooled) сезонный профиль (напр., рынок)."""
    rows = []
    n = len(y)
    for end in range(first_train, n):
        tr = y.iloc[:end]
        base = tr.iloc[-4:].mean()
        own = tr.iloc[-8:].groupby(tr.index[-8:].quarter).mean()
        own = own / own.mean()
        if season_ref is not None:
            r = season_ref.iloc[:end].iloc[-8:]
            pool = r.groupby(r.index.quarter).mean()
            pool = pool / pool.mean()
        for h in range(1, h_max + 1):
            if end + h - 1 >= n:
                break
            tgt = y.index[end + h - 1]
            f = {"наивный (последний квартал)": tr.iloc[-1],
                 "сезонный наивный (год назад)": tr.iloc[end + h - 1 - 4] if end + h - 1 - 4 < end else np.nan,
                 "среднее 4 кв. (без сезона)": base,
                 "среднее 4 кв. × свой сезон": base * own[tgt.quarter]}
            if season_ref is not None:
                f["среднее 4 кв. × сезон рынка"] = base * pool[tgt.quarter]
            for m, v in f.items():
                rows.append({"origin": y.index[end - 1], "target": tgt, "h": h, "method": m,
                             "abs_log_err": abs(np.log(y[tgt] / v))})
    return pd.DataFrame(rows)
