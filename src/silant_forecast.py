"""Прогнозные модели, rolling-origin валидация, иерархическое согласование и интервалы.

Все модели работают с логарифмом сезонно скорректированного ряда z = log(y + c) − log s_q
(c = 10 для сегментов ≥5,5 т с нулями, иначе 0). Интервалы — совместная бутстреп-симуляция:
для всех рядов на каждом шаге берётся остаток из ОДНОГО и того же исторического квартала,
поэтому сохраняется корреляция ошибок между сегментами; затем каждая траектория согласуется
с иерархией (сумма сегментов = тип, сумма типов = рынок).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TYPES = ["ДВС", "Электро"]
MODELS = ["Наивный", "Сезонный наивный", "ETS", "ARX-ставка", "Комбинация"]
BENCH = "Наивный"


# ---------------------------------------------------------------- иерархия
def hierarchy(cols: list[str]) -> tuple[list[str], list[str], pd.DataFrame]:
    """Build the summing matrix: bottom segments -> type (ICE/electric) -> market."""
    bottom = [c for c in cols if any(c.startswith(t + " ") for t in TYPES)]
    upper = TYPES + ["Рынок"]
    nodes = upper + bottom
    Smat = pd.DataFrame(0.0, index=nodes, columns=bottom)
    for b in bottom:
        Smat.loc[b, b] = 1
        Smat.loc[b.split(" ")[0], b] = 1
        Smat.loc["Рынок", b] = 1
    return nodes, bottom, Smat


def offset(name: str) -> float:
    """Constant added before log for series with zeros (segments of 5.5 t and above)."""
    return 10.0 if "5,5" in name else 0.0


# ---------------------------------------------------------------- сезонность (только на обучающих данных)
def season_profile(y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """log-сезонные коэффициенты (центрированы, сумма 0) и их SE: log y ~ эффекты года + дамми кварталов.
    Эффекты года поглощают различия уровня между годами (в т.ч. тренд), остаётся профиль внутри года."""
    import statsmodels.api as sm
    idx = y.index
    X = pd.get_dummies(idx.year).astype(float)
    for k in (2, 3, 4):
        X[f"Q{k}"] = (idx.quarter == k).astype(float)
    X.index = idx
    r = sm.OLS(np.log(y), X).fit()
    b = np.r_[0.0, r.params[["Q2", "Q3", "Q4"]].to_numpy()]
    V = np.zeros((4, 4))
    V[1:, 1:] = r.cov_params().loc[["Q2", "Q3", "Q4"], ["Q2", "Q3", "Q4"]].to_numpy()
    C = np.eye(4) - 0.25
    return C @ b, np.sqrt(np.clip(np.diag(C @ V @ C.T), 0, None))


# ---------------------------------------------------------------- модели на z (log SA)
def ses_fit(z: np.ndarray):
    """Fit simple exponential smoothing on z; returns alpha, last level and one-step residuals."""
    """ETS(A,N,N): простое экспоненциальное сглаживание, α по сетке (минимум SSE одношаговых ошибок)."""
    best = None
    for a in np.linspace(0.05, 1.0, 20):
        l, e = z[0], []
        for v in z[1:]:
            e.append(v - l)
            l = l + a * (v - l)
        sse = float(np.sum(np.square(e)))
        if best is None or sse < best[0]:
            best = (sse, a, l, np.array(e))
    _, a, l, e = best
    return {"alpha": a, "level": l, "resid": e}


def arx_ols(z: pd.Series, x: pd.Series):
    """OLS of z on a constant and exogenous x (aligned to z); returns params, std errors, residuals."""
    import statsmodels.api as sm
    X = sm.add_constant(x.reindex(z.index).to_numpy())
    r = sm.OLS(z.to_numpy(), X).fit()
    return r.params, r.bse, r.resid


def shrink(b: np.ndarray, se: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Эмпирический Байес (DerSimonian–Laird): сжатие коэффициентов к общему среднему группы."""
    w = 1 / se ** 2
    mu_fe = (w * b).sum() / w.sum()
    Q = (w * (b - mu_fe) ** 2).sum()
    t2 = max(0.0, (Q - (len(b) - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    w2 = 1 / (se ** 2 + t2)
    mu = (w2 * b).sum() / w2.sum()
    B = se ** 2 / (se ** 2 + t2) if t2 > 0 else np.ones_like(b)
    return B * mu + (1 - B) * b, mu, np.sqrt(t2)


# ---------------------------------------------------------------- основной прогон
def forecast_all(S: pd.DataFrame, kr: pd.Series, origin: pd.Period, H: int = 4, n_sim: int = 2000,
                 lag: int = 2, seed: int = 42, reconcile: bool = True, infl: dict | None = None,
                 season_from: str = "Электро", sign_constraint: bool = True,
                 inv: pd.Series | None = None, inv_lag: int = 0, kr_pass: float = 1.0,
                 factor_sd: dict | None = None, horizon_infl: np.ndarray | None = None,
                 mixture: bool = False, full_param_se: bool = False) -> dict:
    """Forecast all hierarchy nodes with all models from one origin (bootstrap paths, optional reconciliation).

    Дополнительные слои неопределённости (для горизонта > 4 кв.):
    factor_sd     — {'kr': sd(H), 'inv': sd(H)}: разброс будущих значений факторов (п.п.) вокруг сценарной
                    траектории; шум накапливается как случайное блуждание, стандартное отклонение задаётся по h;
    horizon_infl  — массив (H,) множителей ширины сверх infl (запас на непроверенный горизонт);
    mixture       — добавить модель «Смесь»: каждая траектория целиком берётся из одной из
                    моделей Наивный / ETS / ARX (p = 1/3) — структурная неопределённость «отскок или нет»;
    full_param_se — полная (а не половинная) SE коэффициентов ARX.

    Прогноз всех узлов иерархии всеми моделями из точки origin (данные по origin включительно).
    kr — ключевая ставка по кварталам (нужна до origin+H−lag). Возвращает симуляции и сводки."""
    rng = np.random.default_rng(seed)
    tr = S.loc[:origin]
    nodes, bottom, Smat = hierarchy(list(S.columns))
    fut = pd.period_range(origin + 1, periods=H, freq="Q")
    qf = fut.quarter.to_numpy() - 1

    # сезонный профиль: общий для всех рядов, оценённый по электро (этап 02: сегменты не отличаются
    # от общего профиля, а сезонность ДВС не идентифицируется из-за слома внутри 2025 года)
    prof = {"общий": season_profile(tr[season_from])}
    grp = {n: "общий" for n in nodes}

    Z, off = {}, {}
    for n in nodes:
        c = offset(n)
        off[n] = c
        ls = prof[grp[n]][0]
        Z[n] = np.log(tr[n] + c) - ls[tr.index.quarter - 1]

    # --- ARX: log-уровень = a + Σ b_j·x_j + AR(1); x = ставка(t−lag) [+ инвестиции г/г (t)].
    # Каждый коэффициент сжимается внутри типа (эмпирический Байес), затем знаковые ограничения:
    # ставка b ≤ 0, инвестиции b ≥ 0 (априорная экономическая логика).
    import statsmodels.api as sm
    regs = {"kr": pd.Series(kr.to_numpy(), index=kr.index + lag)}
    if inv is not None:
        regs["inv"] = pd.Series(inv.to_numpy(), index=inv.index + inv_lag)
    rn = list(regs)
    arx = {}
    for g in [["ДВС"] + [b for b in bottom if b.startswith("ДВС")],
              ["Электро"] + [b for b in bottom if b.startswith("Электро")], ["Рынок"]]:
        est = {}
        for n in g:
            X = sm.add_constant(pd.DataFrame({k: v.reindex(Z[n].index) for k, v in regs.items()}))
            r_ = sm.OLS(Z[n], X).fit()
            est[n] = (r_.params[rn].to_numpy(), r_.bse[rn].to_numpy())
        B = np.array([est[n][0] for n in g])
        SE = np.array([est[n][1] for n in g])
        Bs = np.column_stack([shrink(B[:, j], SE[:, j])[0] for j in range(len(rn))]) if len(g) > 1 else B
        if sign_constraint:
            Bs[:, rn.index("kr")] = np.minimum(Bs[:, rn.index("kr")], 0.0)
            if "inv" in rn:
                Bs[:, rn.index("inv")] = np.maximum(Bs[:, rn.index("inv")], 0.0)
        for n, bb, se_ in zip(g, Bs, SE):
            z = Z[n]
            XX = np.column_stack([regs[k].reindex(z.index).to_numpy() for k in rn])
            a = float(np.mean(z.to_numpy() - XX @ bb))
            res = z.to_numpy() - a - XX @ bb
            phi = float(np.clip(np.corrcoef(res[1:], res[:-1])[0, 1], 0, 0.9)) if len(res) > 3 else 0.0
            innov = res[1:] - phi * res[:-1]
            innov = innov - innov.mean()
            arx[n] = {"a": a, "b": bb, "se_b": se_, "names": rn, "phi": phi, "last_res": res[-1], "innov": innov}

    ets = {n: ses_fit(Z[n].to_numpy()) for n in nodes}

    # общий набор «исторических кварталов» для совместного бутстрепа
    T = len(tr)
    XF = np.column_stack([regs[k].reindex(fut).to_numpy() for k in rn])   # (H, k)
    if kr_pass != 1.0:
        # неполная передача изменения ставки в спрос: эффективная ставка = ставка на последнем
        # наблюдаемом квартале + kr_pass · (сценарная ставка − она)
        ref = regs["kr"][origin]
        XF[:, rn.index("kr")] = ref + kr_pass * (XF[:, rn.index("kr")] - ref)
    if np.isnan(XF).any():
        raise ValueError(f"нет значений факторов {rn} на {list(fut[np.isnan(XF).any(1)])}")

    models = MODELS + (["Смесь"] if mixture else [])
    draws = {m: {n: np.empty((n_sim, H)) for n in nodes} for m in models}
    XFd = np.broadcast_to(XF, (n_sim,) + XF.shape).copy()            # (n_sim, H, k)
    if factor_sd:
        frng = np.random.default_rng(seed + 1000)
        for k_, sd in factor_sd.items():
            if k_ in rn:
                sd = np.asarray(sd, float)
                # случайное блуждание с заданным sd на каждом h: приращения ~ N(0, sd_h² − sd_{h−1}²)
                inc_sd = np.sqrt(np.clip(np.diff(np.r_[0.0, sd ** 2]), 0, None))
                XFd[:, :, rn.index(k_)] += np.cumsum(frng.normal(size=(n_sim, H)) * inc_sd, axis=1)
    idx_d = rng.integers(0, T - 1, size=(n_sim, H))       # индекс для одношаговых остатков (длина T−1)
    idx_s = rng.integers(0, T - 4, size=(n_sim, H))       # для сезонных разностей (длина T−4)
    seas_noise = {g: rng.normal(size=(n_sim, 4)) * prof[g][1] for g in prof}
    bnoise = rng.normal(size=(n_sim, len(rn)))
    mix_pick = np.random.default_rng(seed + 2000).integers(0, 3, size=n_sim)

    for n in nodes:
        z = Z[n].to_numpy()
        c = off[n]
        ls = prof[grp[n]][0][qf] + seas_noise[grp[n]][:, qf]          # (n_sim, H)
        # Наивный: случайное блуждание по z
        dz = np.diff(z)
        dz = dz - dz.mean()          # без дрейфа
        path = z[-1] + np.cumsum(dz[idx_d], axis=1)
        draws["Наивный"][n] = np.exp(path + ls) - c
        # Сезонный наивный: y_{T+h} = y_{T+h-4} · exp(ошибка сезонной разности)
        ly = np.log(tr[n].to_numpy() + c)
        sd = ly[4:] - ly[:-4]
        base = ly[T - 4 + (np.arange(H) % 4)]
        draws["Сезонный наивный"][n] = np.exp(base + (sd - sd.mean())[idx_s]) - c
        # ETS(A,N,N)
        e = ets[n]
        inn = (e["resid"] - e["resid"].mean())[idx_d]
        lvl = e["level"] + np.cumsum(np.c_[np.zeros(n_sim), e["alpha"] * inn[:, :-1]], axis=1)
        draws["ETS"][n] = np.exp(lvl + inn + ls) - c
        # ARX: a + b·ставка(t−lag) + AR(1)-ошибка; неопределённость b
        p = arx[n]
        bb = p["b"][None, :] + bnoise * p["se_b"][None, :] * (1.0 if full_param_se else 0.5)   # половина SE: b уже сжат к среднему группы
        if sign_constraint and full_param_se:   # при полной SE знаковые ограничения соблюдаются и в розыгрышах
            bb[:, rn.index("kr")] = np.minimum(bb[:, rn.index("kr")], 0)
            if "inv" in rn:
                bb[:, rn.index("inv")] = np.maximum(bb[:, rn.index("inv")], 0)
        innv = p["innov"][np.clip(idx_d, 0, len(p["innov"]) - 1)]
        r = np.empty((n_sim, H))
        prev = np.full(n_sim, p["last_res"])
        for h in range(H):
            prev = p["phi"] * prev + innv[:, h]
            r[:, h] = prev
        draws["ARX-ставка"][n] = np.exp(p["a"] + np.einsum("sk,shk->sh", bb, XFd) + r + ls) - c
        # Комбинация: среднее логарифмов Наивной, ETS и ARX (одни и те же розыгрыши)
        draws["Комбинация"][n] = np.exp((np.log(draws["Наивный"][n] + c) + np.log(draws["ETS"][n] + c) + np.log(draws["ARX-ставка"][n] + c)) / 3) - c

    # --- калибровка ширины интервалов: растяжение лог-отклонений от медианы в k раз
    if infl or horizon_infl is not None:
        infl = dict(infl or {})
        hk = np.ones(H) if horizon_infl is None else np.asarray(horizon_infl, float)
        for m in models:
            if m == "Смесь":
                continue   # смесь собирается из уже калиброванных моделей-компонент (ниже)
            k = infl.get(m, 1.0) * hk[None, :]
            for n in nodes:
                c = off[n]
                lg = np.log(np.clip(draws[m][n], 0, None) + c + 1e-9)
                med = np.median(lg, axis=0, keepdims=True)
                draws[m][n] = np.exp(med + k * (lg - med)) - c

    if mixture:
        pick = mix_pick[:, None]
        for n in nodes:
            draws["Смесь"][n] = np.where(pick == 0, draws["Наивный"][n], np.where(pick == 1, draws["ETS"][n], draws["ARX-ставка"][n]))

    # --- согласование (WLS, веса = дисперсия одношаговых ошибок на уровне последнего значения)
    rec = {}
    for m in models:
        if not reconcile:
            rec[m] = draws[m]
            continue
        sig = np.array([np.std(np.diff(Z[n].to_numpy())) * (tr[n].iloc[-4:].mean() + 1) for n in nodes])
        w_ = 1 / sig ** 2
        Sm = Smat.to_numpy()
        # weights are applied by broadcasting/einsum instead of a dense diag matrix: the zeros of the
        # summing matrix made the macOS BLAS matmul raise a spurious "invalid value" RuntimeWarning
        # (all inputs and outputs are finite, the result is identical up to 1e-16)
        G = np.linalg.solve(np.einsum("nb,n,nc->bc", Sm, w_, Sm), Sm.T * w_[None, :])       # bottom = G · base
        stack = np.stack([np.clip(draws[m][n], 0, None) for n in nodes], axis=-1)   # (n_sim,H,nodes)
        bot = np.einsum("bn,shn->shb", G, stack)
        bot = np.clip(bot, 0, None)
        full = np.einsum("nb,shb->shn", Sm, bot)
        rec[m] = {n: full[:, :, i] for i, n in enumerate(nodes)}

    return {"fut": fut, "nodes": nodes, "bottom": bottom, "base": draws, "rec": rec,
            "arx": arx, "ets": ets, "prof": prof}


def summarize(sims: dict, fut, qs=(0.025, 0.1, 0.5, 0.9, 0.975)) -> pd.DataFrame:
    """Mean and quantiles of simulated paths per node and forecast quarter."""
    rows = []
    for n, d in sims.items():
        for h, q in enumerate(fut):
            v = d[:, h]
            rows.append({"node": n, "quarter": str(q), "h": h + 1, "mean": v.mean(),
                         **{f"q{int(p * 1000):03d}": np.quantile(v, p) for p in qs}})
    return pd.DataFrame(rows)


def share_draws(sims: dict) -> np.ndarray:
    """Electric share of the market for each simulated path and horizon."""
    return sims["Электро"] / (sims["Электро"] + sims["ДВС"])


# ---------------------------------------------------------------- rolling-origin
def persist_exog(x: pd.Series, origin: pd.Period, horizon: int) -> pd.Series:
    """Keep x only up to origin; later quarters get the last value known at origin (persistence).

    x is indexed by the quarter of observation (before any lag is applied), so the lag is
    handled by forecast_all: the lagged regressor is known up to origin + lag."""
    known = x.loc[:origin].dropna()
    idx = pd.period_range(x.index.min(), origin + horizon, freq="Q")
    out = x.reindex(idx)
    out.loc[out.index > origin] = known.iloc[-1]
    return out


def rolling_origin(S: pd.DataFrame, kr: pd.Series, first: pd.Period, last: pd.Period, H=4,
                   n_sim=800, seed=42, reconcile=True, infl=None, persist_exogenous=True, **kw) -> pd.DataFrame:
    """Rolling-origin backtest: forecast H quarters from every origin in [first, last] and score against actuals.

    persist_exogenous=True (default): at each origin the exogenous rate (and investment, if passed)
    is cut at the origin and held at its last known value, so the backtest does not see the future
    rate. persist_exogenous=False reproduces the old oracle backtest that used the actual future rate."""
    rows = []
    for k, o in enumerate(pd.period_range(first, last, freq="Q")):
        kr_o, kw_o = kr, kw
        if persist_exogenous:
            kr_o = persist_exog(kr, o, H)
            if kw.get("inv") is not None:
                kw_o = {**kw, "inv": persist_exog(kw["inv"], o, H)}
        fc = forecast_all(S, kr_o, o, H=H, n_sim=n_sim, seed=seed + k, reconcile=reconcile, infl=infl, **kw_o)
        for src in (["base", "rec"] if reconcile else ["base"]):
            for m in fc[src]:
                sims = fc[src][m]
                for n in fc["nodes"] + ["Доля электро"]:
                    d = share_draws(sims) if n == "Доля электро" else sims[n]
                    for h, q in enumerate(fc["fut"]):
                        if q not in S.index:
                            continue
                        act = (S.loc[q, "Электро"] / (S.loc[q, "Электро"] + S.loc[q, "ДВС"])
                               if n == "Доля электро" else S.loc[q, n])
                        v = d[:, h]
                        # точка для метрики — среднее траекторий (в отчётах прогноз — медиана);
                        # проверено: с медианой ошибка и смещение меньше на 1–3 п.п., выводы не меняются
                        pt = v.mean()
                        rows.append({"origin": str(o), "target": str(q), "h": h + 1, "src": src, "model": m,
                                     "node": n, "actual": act, "point": pt,
                                     "abs_log_err": abs(np.log((act + offset(n)) / (pt + offset(n)))) if n != "Доля электро" else abs(act - pt),
                                     "in80": np.quantile(v, .1) <= act <= np.quantile(v, .9),
                                     "in95": np.quantile(v, .025) <= act <= np.quantile(v, .975)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- уровень плана при асимметрии потерь
def critical_ratio(cu: float, co: float) -> float:
    """Критический квантиль задачи «газетчика»: α* = Cu / (Cu + Co).

    cu — потеря от недостачи одной единицы (упущенная маржинальная прибыль),
    co — потеря от излишка одной единицы (деньги в запасе, хранение, скидка при распродаже).
    План = α*-квантиль распределения спроса: при таком уровне ожидаемые потери минимальны."""
    if cu < 0 or co < 0 or cu + co == 0:
        raise ValueError("потери должны быть неотрицательными и не обе нулевыми")
    return cu / (cu + co)


def plan_multipliers(sums: dict, alphas) -> pd.DataFrame:
    """Множители квантилей к медиане: m(α) = Q_α / Q_0,5 распределения суммы за горизонт.

    sums — {ряд: массив траекторий суммы за горизонт}. Множитель переносит форму распределения
    модели на любой центр (например, на уровень «без изменения условий»)."""
    rows = {}
    for n, v in sums.items():
        med = np.quantile(v, 0.5)
        rows[n] = {round(float(a), 4): float(np.quantile(v, a) / med) for a in alphas}
    return pd.DataFrame(rows).T
