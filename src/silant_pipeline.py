"""Инструмент обновления: проверка нового файла → пересчёт ноутбуков → отчёты → архив запуска."""
from __future__ import annotations

import base64
import datetime as dt
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from silant_data import check_formula_ranges, check_totals, load_market

RAW_NAME = "silant_forklift_market.xlsx"
NOTEBOOKS = ["01_data_eda", "02_trend_seasonality", "03_external_factors", "04_forecast", "05_scenarios", "06_long_horizon"]
KEY = ["Рынок", "ДВС", "Электро"]


# ---------------------------------------------------------------- 1. проверка файла рынка
def validate_market(path: Path, prev_long: pd.DataFrame | None = None) -> dict:
    """Проверки нового Excel. errors — останавливают обновление, warnings — только сообщаются."""
    errors, warnings, info = [], [], []
    try:
        long, totals, formulas = load_market(path)
    except Exception as e:  # noqa: BLE001
        return {"errors": [f"Файл не читается парсером: {e}"], "warnings": [], "info": [], "long": None}

    q = long.quarter.drop_duplicates().sort_values()
    info.append(f"Кварталы в файле: {q.min()} – {q.max()} ({len(q)} шт.)")
    full = pd.period_range(q.min(), q.max(), freq="Q")
    if len(full) != len(q):
        errors.append(f"Пропущены кварталы: {sorted(set(full) - set(q))}")
    if long.units.isna().any():
        bad = long[long.units.isna()][["type", "segment", "quarter"]].astype(str).values.tolist()[:10]
        errors.append(f"Пустые или нечисловые значения: {len(long[long.units.isna()])} ячеек, например {bad}")
    if (long.units < 0).any():
        errors.append("Есть отрицательные значения")
    seg = long.groupby("type").segment.nunique().to_dict()
    info.append(f"Сегментов: {seg}")

    tot = check_totals(long, totals)
    if tot.total_file.isna().any():
        qs = sorted({x for x in tot[tot.total_file.isna()].quarter})
        warnings.append(f"В строке «Итого» нет сохранённых значений ({len(qs)} кв.: {qs[0]}…{qs[-1]}) — "
                        "файл сохранён программой, которая не пересчитала формулы (откройте и сохраните в Excel). "
                        "Сверка с «Итого» пропущена; итоги пайплайн считает из сегментов")
    badt = tot[tot.total_file.notna() & ~tot.ok]
    if len(badt):
        errors.append("Сумма сегментов ≠ «Итого»: " + "; ".join(
            f"{r.type} {r.quarter}: сегменты {r.sum_segments:.0f}, итого {r.total_file}" for r in badt.itertuples()))
    fr = check_formula_ranges(long, formulas)
    for r in fr[~fr.exact_match].itertuples():
        if r.header_only_extra:
            info.append(f"Формула «Итого» ({r.type}) {r.formula} захватывает строки заголовка блока (сегменты — {r.segment_rows}); "
                        "текст SUM не суммирует, на итог не влияет")
        else:
            warnings.append(f"Формула «Итого» ({r.type}) {r.formula} охватывает строки {r.formula_rows}, сегменты — {r.segment_rows}")
    if not (long.units.dropna() % 10 == 0).all():
        warnings.append("Не все значения кратны 10 (раньше были кратны) — проверьте единицы")

    new_q, revisions = [], pd.DataFrame()
    if prev_long is not None and len(prev_long):
        pl = prev_long.copy()
        pl["quarter"] = pd.PeriodIndex(pl.quarter.astype(str), freq="Q")
        prev_q = set(pl.quarter)
        new_q = sorted(set(q) - prev_q)
        lost = sorted(prev_q - set(q))
        if lost:
            errors.append(f"В новом файле пропали кварталы, которые были раньше: {lost}")
        m = long.merge(pl[["type", "segment", "quarter", "units"]], on=["type", "segment", "quarter"],
                       how="inner", suffixes=("", "_prev"))
        revisions = m[m.units != m.units_prev][["type", "segment", "quarter", "units_prev", "units"]]
        if len(revisions):
            warnings.append(f"Пересмотрены исторические значения: {len(revisions)} ячеек (см. лист «Проверки»)")
        seg_prev = set(zip(pl.type, pl.segment))
        seg_now = set(zip(long.type, long.segment))
        if seg_prev != seg_now:
            warnings.append(f"Изменился состав сегментов: добавлены {seg_now - seg_prev}, убраны {seg_prev - seg_now}")
        info.append("Новые кварталы: " + (", ".join(map(str, new_q)) if new_q else "нет (пересчёт на тех же данных)"))
    # аномальные скачки в новых кварталах: отклонение от того же квартала год назад
    for nq in new_q:
        d = long[long.quarter == nq].set_index(["type", "segment"]).units
        p = long[long.quarter == nq - 4].set_index(["type", "segment"]).units
        r = (d / p.replace(0, np.nan))
        big = r[(p > 200) & ((r > 3) | (r < 1 / 3))]
        for (t, s), v in big.items():
            warnings.append(f"{nq} {t} {s}: {d[(t, s)]:.0f} шт. против {p[(t, s)]:.0f} год назад (×{v:.1f}) — проверьте ввод")
    return {"errors": errors, "warnings": warnings, "info": info, "long": long, "new_quarters": new_q,
            "revisions": revisions, "last": q.max()}


def check_external(root: Path, last: pd.Period) -> list[str]:
    """Покрытие внешних данных относительно последнего квартала рынка."""
    import sys
    sys.path.insert(0, str(root / "src"))
    from silant_macro import RETRIEVED, build_macro
    from silant_scenarios import scenario_paths
    msgs = []
    _, W = build_macro(root)
    for col, name in [("key_rate_avg", "ключевая ставка"), ("invest_fc_yoy", "инвестиции в основной капитал (Росстат)")]:
        lq = W[col].dropna().index.max()
        if lq < last:
            msgs.append(f"{name}: последний факт {lq}, а рынок уже до {last} — для недостающих кварталов "
                        f"будут использованы сценарные допущения. Обновите источник (см. data/external/SOURCES.md)")
    age = (pd.Timestamp.today() - pd.Timestamp(RETRIEVED)).days
    if age > 60:
        msgs.append(f"Снимок данных ЦБ от {RETRIEVED} ({age} дн. назад) — запустите с --fetch при доступе к cbr.ru")
    _, _, notes = scenario_paths(W, root, last, H=8)
    msgs += notes
    return msgs


# ---------------------------------------------------------------- 2. запуск ноутбуков
def run_notebooks(root: Path, names=NOTEBOOKS, log=print) -> list[dict]:
    """Regenerate each notebook from build_0X.py, execute it, and save the notebook and an HTML copy."""
    import subprocess
    import sys

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
    from nbconvert import HTMLExporter

    nbdir, rep = root / "notebooks", root / "reports"
    rep.mkdir(exist_ok=True)
    res = []
    for name in names:
        t0 = time.time()
        num = name[:2]
        subprocess.run([sys.executable, f"build_{num}.py"], cwd=nbdir, check=True)
        path = nbdir / f"{name}.ipynb"
        nb = nbformat.read(path, as_version=4)
        client = NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(nbdir)}},
                                extra_arguments=["--IPKernelApp.log_level=ERROR"])
        try:
            client.execute()
        except CellExecutionError as e:
            nbformat.write(nb, path)
            msg = str(e).split("\n")
            tail = [l for l in msg if "Error" in l][-1:] or msg[-1:]
            raise RuntimeError(f"Ноутбук {name} упал: {tail[0][:300]}") from None
        nbformat.write(nb, path)
        body, _ = HTMLExporter().from_notebook_node(nb)
        (rep / f"{name}.html").write_text(body, encoding="utf-8")
        warn = [line for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", [])
                if o.get("output_type") == "stream" for line in o.get("text", "").splitlines() if "⚠️" in line]
        res.append({"notebook": name, "sec": round(time.time() - t0, 1), "warnings": warn})
        log(f"  ✓ {name}: {time.time() - t0:.0f} с")
    return res


# ---------------------------------------------------------------- 3. сравнение с прошлым запуском
def latest_run(root: Path) -> Path | None:
    """Return the folder of the last archived run, or None."""
    f = root / "data/runs/LATEST.txt"
    if f.exists():
        p = root / "data/runs" / f.read_text(encoding="utf-8").strip()
        return p if p.exists() else None
    return None


def diff_with_previous(root: Path, prev: Path | None) -> pd.DataFrame:
    """Compare the current 4-quarter forecast with the one from the previous archived run."""
    if prev is None:
        return pd.DataFrame()
    cur = pd.read_csv(root / "data/processed/forecast_4q_total.csv", index_col=0)
    old = pd.read_csv(prev / "forecast_4q_total.csv", index_col=0)
    meta_old = json.loads((prev / "run_meta.json").read_text(encoding="utf-8")) if (prev / "run_meta.json").exists() else {}
    rows = []
    for n in cur.index.intersection(old.index):
        rows.append({"ряд": n, "прошлый запуск: данные до": meta_old.get("last_quarter", "?"),
                     "прошлый прогноз 4 кв., медиана": old.loc[n, "P50"], "новый прогноз 4 кв., медиана": cur.loc[n, "P50"],
                     "прошлый P10": old.loc[n, "P10"], "прошлый P90": old.loc[n, "P90"],
                     "новый P10": cur.loc[n, "P10"], "новый P90": cur.loc[n, "P90"]})
    return pd.DataFrame(rows)


def backtest_new_quarter(root: Path, prev: Path | None, new_q: list) -> pd.DataFrame:
    """Как прошлый прогноз «угадал» новые фактические кварталы (честная проверка вне выборки)."""
    if prev is None or not new_q or not (prev / "forecast_4q.csv").exists():
        return pd.DataFrame()
    fc = pd.read_csv(prev / "forecast_4q.csv")
    long = pd.read_csv(root / "data/processed/market_long.csv")
    import sys
    sys.path.insert(0, str(root / "src"))
    rows = []
    for q in map(str, new_q):
        f = fc[fc.quarter == q]
        if f.empty:
            continue
        act = long[long.quarter == q]
        a = {"Рынок": act.units.sum(), "ДВС": act[act.type == "ДВС"].units.sum(), "Электро": act[act.type == "Электро"].units.sum()}
        for seg, g in act.assign(key=act.type + " " + np.where(act.cap_min >= 5.5, "5,5 т+", act.segment)).groupby("key"):
            a[seg] = g.units.sum()
        for n, v in a.items():
            r = f[f.node == n]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append({"квартал": q, "ряд": n, "факт": v, "прогноз (медиана)": r.q500, "P10": r.q100, "P90": r.q900,
                         "ошибка, %": v / r.q500 - 1, "внутри 80%": bool(r.q100 <= v <= r.q900)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 3а. решающее правило по ДВС
DVS_RULE = {"q2_border": 7_700, "q23_low": 13_500, "q23_high": 16_000}


def dvs_rule(long: pd.DataFrame) -> dict:
    """Решающее правило по ДВС (отчёт, табл. рекомендаций): по факту рынка ДВС во II и III кв. 2026 г.

    Сумма II + III кв. ≤ 13,5 тыс. — спад сохраняется; ≥ 16,0 тыс. — восстановление; между — ждать IV кв.
    Пока есть только II кв.: предварительный сигнал, граница ≈7,7 тыс. (оценки модели: 6,7 тыс. при спаде, 8,7 тыс. при восстановлении);
    как на листе «План_завода» рабочей модели Excel (Параметры!B10).
    Возвращает статус («ожидание», «предварительно», «решение»), вывод, действие и подходящий сценарий спроса."""
    dv = long[long.type == "ДВС"].groupby(long.quarter.astype(str)).units.sum()
    b2, b3, q2, q3 = (dv.get(k) for k in ("2025Q2", "2025Q3", "2026Q2", "2026Q3"))
    R = DVS_RULE
    if q2 is not None and q3 is not None:
        s = q2 + q3
        fact = f"ДВС за II + III кв. 2026 г.: {s:,.0f} шт. ({s / (b2 + b3) - 1:+.0%} к тому же периоду 2025 г.)".replace(",", " ")
        if s <= R["q23_low"]:
            return {"status": "решение", "fact": fact, "verdict": "спад сохраняется",
                    "action": "выпуск ДВС не наращивать, ускорять перевод на электро 3–5 т", "scenario": "Ускоренное замещение"}
        if s >= R["q23_high"]:
            return {"status": "решение", "fact": fact, "verdict": "восстановление",
                    "action": "можно наращивать ДВС с IV кв. 2026 г.", "scenario": "Восстановление ДВС вслед за ставкой"}
        return {"status": "решение", "fact": fact, "verdict": "между границами (13,5–16,0 тыс.)",
                "action": "держать базовый план, решение — по факту IV кв.", "scenario": "Базовый макросценарий (комбинация)"}
    if q2 is not None:
        fact = f"ДВС во II кв. 2026 г.: {q2:,.0f} шт. ({q2 / b2 - 1:+.0%} к II кв. 2025 г.)".replace(",", " ")
        if q2 >= R["q2_border"]:
            v, a, sc = ("предварительно: признак восстановления", "решение — после III кв., по сумме II + III кв.",
                        "Восстановление ДВС вслед за ставкой")
        else:
            v, a, sc = "предварительно: спад сохраняется", "решение — после III кв., по сумме II + III кв.", "Спад сохраняется"
        return {"status": "предварительно", "fact": fact, "verdict": v, "action": a, "scenario": sc}
    return {"status": "ожидание", "fact": "Факта за II кв. 2026 г. пока нет", "verdict": "ожидание факта",
            "action": "сумма II + III кв. ≤ 13,5 тыс. (+23 % г/г) — спад, ≥ 16,0 тыс. (+45 %) — восстановление, между — ждать IV кв.; "
                      "II кв. — только предварительный сигнал (граница ≈7,7 тыс.)",
            "scenario": "Спад сохраняется"}


# ---------------------------------------------------------------- 4. Excel-отчёт
def build_excel(root: Path, out: Path, meta: dict, checks: dict, diff: pd.DataFrame, oos: pd.DataFrame):
    """Write the Excel workbook with the key result tables, checks and changes since the last run."""
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    P = root / "data/processed"
    F = "Arial"
    HFILL = PatternFill("solid", fgColor="E8E6DF")
    thin = Side(style="thin", color="C3C2B7")
    wb = Workbook()

    def sheet(title, df, note=None, num_fmt=None, pct_cols=(), int_cols=(), widths=None, first=False):
        ws = wb.active if first else wb.create_sheet()
        ws.title = title[:31]
        r0 = 1
        if note:
            ws.cell(1, 1, note).font = Font(name=F, italic=True, size=9, color="52514E")
            r0 = 3
        for j, c in enumerate(df.columns, 1):
            cell = ws.cell(r0, j, str(c))
            cell.font = Font(name=F, bold=True, size=10)
            cell.fill = HFILL
            cell.alignment = Alignment(wrap_text=True, vertical="top")
            cell.border = Border(bottom=thin)
        for i, row in enumerate(df.itertuples(index=False), r0 + 1):
            for j, v in enumerate(row, 1):
                if isinstance(v, (np.floating, float)) and np.isnan(v):
                    v = None
                elif isinstance(v, np.generic):
                    v = v.item()
                cell = ws.cell(i, j, v)
                cell.font = Font(name=F, size=10)
                col = df.columns[j - 1]
                if col in pct_cols:
                    cell.number_format = "0.0%"
                elif col in int_cols or (num_fmt and isinstance(v, (int, float))):
                    cell.number_format = num_fmt or "#,##0"
        for j, c in enumerate(df.columns, 1):
            w = (widths or {}).get(c) or min(max(len(str(c)), *(len(str(x)) for x in df.iloc[:, j - 1].head(50)), 8) + 2, 60)
            ws.column_dimensions[get_column_letter(j)].width = w
        ws.freeze_panes = ws.cell(r0 + 1, 1)
        return ws, r0

    IND_RU = {
        "cny_rub_avg": "Курс CNY/RUB, средний", "usd_rub_avg": "Курс USD/RUB, средний", "usd_rub_end": "Курс USD/RUB, на конец квартала",
        "credit_debt_end_bln": "Задолженность юрлиц по кредитам, на конец квартала", "credit_debt_yoy": "Задолженность юрлиц по кредитам, г/г",
        "credit_new_bln": "Выдачи кредитов юрлицам", "credit_new_yoy": "Выдачи кредитов юрлицам, г/г",
        "goods_imports_musd": "Импорт товаров (платёжный баланс)", "goods_imports_yoy": "Импорт товаров, г/г",
        "inv_act_econ": "Инвестактивность: экономика в целом", "inv_act_industry": "Инвестактивность: промышленность",
        "inv_act_transport_storage": "Инвестактивность: транспорт и хранение", "inv_act_wholesale": "Инвестактивность: оптовая торговля",
        "inv_exp_econ": "Ожидания инвестактивности: экономика в целом", "inv_exp_industry": "Ожидания инвестактивности: промышленность",
        "inv_exp_transport_storage": "Ожидания инвестактивности: транспорт и хранение", "inv_exp_wholesale": "Ожидания инвестактивности: оптовая торговля",
        "invest_fc_yoy": "Инвестиции в основной капитал, г/г", "ipp_yoy": "Промышленное производство, г/г",
        "key_rate_avg": "Ключевая ставка, средняя за квартал", "key_rate_end": "Ключевая ставка, на конец квартала",
        "rate_passthrough": "Передача снижения ставки в спрос, доля",
    }
    
    # --- Сводка
    tot = pd.read_csv(P / "forecast_4q_total.csv", index_col=0)
    order = KEY + [n for n in tot.index if n not in KEY]
    s1 = tot.loc[order, ["факт: последние 4 кв.", "P10", "P50", "P90"]].reset_index().rename(columns={"index": "ряд"})
    s1.columns = ["ряд", "факт: последние 4 кв., шт.", "P10, шт.", "медиана, шт.", "P90, шт."]
    ws, r0 = sheet("Сводка", s1, note=(f"Прогноз на 4 квартала ({meta['horizon_4q']}) по данным до {meta['last_quarter']}. "
                                       f"Модель «Комбинация» (Наивный+ETS+ARX), согласованная. Запуск {meta['run_time']}."),
                   num_fmt="#,##0", first=True)
    ws.cell(r0, 6, "изм. медианы к факту").font = Font(name=F, bold=True, size=10)
    ws.cell(r0, 6).fill = HFILL
    for i in range(r0 + 1, r0 + 1 + len(s1)):
        c = ws.cell(i, 6, f"=IF(B{i}=0,\"-\",D{i}/B{i}-1)")
        c.number_format = "+0%;-0%;0%"
        c.font = Font(name=F, size=10)
    ws.column_dimensions["F"].width = 20
    ws.cell(r0 + 1, 6).comment = Comment("Формула: медиана прогноза / факт за последние 4 квартала − 1", "pipeline")
    # доля электро
    sh = pd.read_csv(P / "forecast_4q_e_share.csv")
    rr = r0 + len(s1) + 3
    ws.cell(rr, 1, "Доля электропогрузчиков (прогноз по кварталам)").font = Font(name=F, bold=True)
    for j, c in enumerate(sh.columns, 1):
        ws.cell(rr + 1, j, c).font = Font(name=F, bold=True, size=10)
        ws.cell(rr + 1, j).fill = HFILL
    for i, row in enumerate(sh.itertuples(index=False), rr + 2):
        for j, v in enumerate(row, 1):
            c = ws.cell(i, j, v if not isinstance(v, np.generic) else v.item())
            c.font = Font(name=F, size=10)
            if j > 1:
                c.number_format = "0%"

    # --- Уровень плана при асимметрии потерь (этап 04, раздел 5.7)
    if (P / "forecast_4q_plan_levels.csv").exists():
        pl = pd.read_csv(P / "forecast_4q_plan_levels.csv")
        sheet("Уровень плана", pl, note=("План = квантиль спроса α = Cu/(Cu+Co): Cu — потеря от недостачи единицы, Co — от излишка. "
                                         "Сумма за 4 квартала, шт. рынка; для завода умножить на долю. Расчёт со своими Cu и Co — "
                                         "рабочая модель Excel, лист «Уровень плана»."), num_fmt="#,##0")

    # решающее правило по ДВС
    rule = dvs_rule(pd.read_csv(P / "market_long.csv"))
    rr2 = rr + len(sh) + 3
    ws.cell(rr2, 1, "Решающее правило по ДВС").font = Font(name=F, bold=True)
    for i, (k, v) in enumerate([("факт", rule["fact"]), ("вывод", rule["verdict"]), ("действие", rule["action"]),
                                ("сценарий спроса", rule["scenario"])], rr2 + 1):
        ws.cell(i, 1, k).font = Font(name=F, size=10, color="52514E")
        ws.cell(i, 2, v).font = Font(name=F, size=10, bold=(k == "вывод"))

    # --- Прогноз по кварталам
    q = pd.read_csv(P / "forecast_4q.csv")
    q = q[["node", "quarter", "q025", "q100", "q500", "q900", "q975", "mean"]]
    q.columns = ["ряд", "квартал", "P2.5", "P10", "медиана", "P90", "P97.5", "среднее (аддитивно)"]
    sheet("Прогноз 4 кв по кварталам", q, note="Интервалы откалиброваны по скользящей проверке на истории (этап 04). Медианы сегментов не складываются точно в итог — аддитивны средние.",
          num_fmt="#,##0")

    # --- Сценарии
    sc = pd.read_csv(P / "long_forecast_totals_layers.csv")
    pvt = sc.pivot_table(index=["ряд", "период", "сценарий"], columns="слой", values=["P10", "P50", "P90"]).reset_index()
    pvt.columns = [" ".join([a, b]).strip() for a, b in pvt.columns]
    keep = ["ряд", "период", "сценарий", "P50 L1", "P10 L1", "P90 L1", "P10 L3", "P90 L3", "P10 L4", "P90 L4"]
    pvt = pvt[keep].rename(columns={"P50 L1": "медиана (центр)", "P10 L1": "P10 L1 статист.", "P90 L1": "P90 L1 статист.",
                                    "P10 L3": "P10 L3 +горизонт+факторы", "P90 L3": "P90 L3 +горизонт+факторы",
                                    "P10 L4": "P10 L4 +структурный", "P90 L4": "P90 L4 +структурный"})
    pvt["_o"] = pvt["ряд"].map({k: i for i, k in enumerate(order)})
    pvt = pvt.sort_values(["_o", "период", "сценарий"]).drop(columns="_o")
    sheet("Сценарии 8 кв", pvt, note="Суммы за 4 квартала. Слои интервалов — этап 06; сценарные допущения — лист «Допущения».", num_fmt="#,##0")
    ass = pd.read_csv(root / "data/external/manual/scenarios.csv")
    ass = ass.assign(factor=ass["factor"].map(IND_RU).fillna(ass["factor"])).rename(
        columns={"scenario": "сценарий", "factor": "фактор", "quarter": "квартал", "value": "значение", "basis": "обоснование"})
    sheet("Допущения", ass, note="Редактируемый источник: data/external/manual/scenarios.csv", widths={"обоснование": 110})

    # --- Факт
    ml = pd.read_csv(P / "market_long.csv")
    fw = ml.pivot_table(index=["type", "segment"], columns="quarter", values="units", aggfunc="sum", sort=False).reset_index()
    fw = fw.rename(columns={"type": "тип", "segment": "сегмент"})
    sheet("Факт рынка", fw, note="Данные заказчика по сегментам и кварталам, шт.", num_fmt="#,##0")
    mw = pd.read_csv(P / "macro_quarterly_wide.csv")
    mw = mw.rename(columns={"quarter": "квартал", **IND_RU})
    sheet("Внешние факторы", mw, note="Квартальные ряды; единицы, источники и ссылки — лист «Источники»", num_fmt="0.00")
    msrc = pd.read_csv(P / "macro_quarterly_long.csv")[["indicator", "unit", "source_org", "publisher", "url", "retrieved"]].drop_duplicates("indicator")
    msrc = msrc.assign(indicator=msrc["indicator"].map(IND_RU).fillna(msrc["indicator"])).rename(columns={
        "indicator": "показатель", "unit": "единица", "source_org": "источник", "publisher": "где опубликовано", "url": "ссылка", "retrieved": "дата выгрузки"})
    sheet("Источники", msrc, widths={"ссылка": 80})

    # --- Валидация
    bt = pd.read_csv(P / "backtest_rolling_origin.csv")
    b = bt[(bt.src == "rec") & bt.node.isin(KEY)].pivot_table(index=["model", "h"], columns="node", values="abs_log_err").reset_index()
    b = b.rename(columns={"model": "модель", "h": "горизонт, кв."})
    sheet("Валидация", b, note="Средняя |log(факт/прогноз)| при скользящей проверке на истории по горизонту прогноза (≈ ошибка в %)", pct_cols=tuple(KEY))

    # --- Проверки и изменения
    rows = [("ошибка", e) for e in checks.get("errors", [])] + [("предупреждение", w) for w in checks.get("warnings", [])] \
        + [("информация", i) for i in checks.get("info", [])] + [("внешние данные", m) for m in checks.get("external", [])]
    sheet("Проверки", pd.DataFrame(rows, columns=["тип", "сообщение"]) if rows else pd.DataFrame({"тип": ["ок"], "сообщение": ["Замечаний нет"]}),
          widths={"сообщение": 140})
    rev = checks.get("revisions")
    if rev is not None and len(rev):
        sheet("Пересмотры истории", rev.astype({"quarter": str}), num_fmt="#,##0")
    if len(diff):
        ws, r0 = sheet("Изменения к прошлому", diff, note="Сравнение прогноза на 4 квартала с прошлым запуском (горизонты могут отличаться, если добавлен квартал)",
                       num_fmt="#,##0")
        col = len(diff.columns) + 1
        ws.cell(r0, col, "изм. медианы").font = Font(name=F, bold=True, size=10)
        ws.cell(r0, col).fill = HFILL
        for i in range(r0 + 1, r0 + 1 + len(diff)):
            c = ws.cell(i, col, f"=IF(C{i}=0,\"-\",D{i}/C{i}-1)")
            c.number_format = "+0%;-0%;0%"
            c.font = Font(name=F, size=10)
    if len(oos):
        sheet("Проверка на новом факте", oos, note="Прогноз прошлого запуска против фактических значений добавленного квартала",
              num_fmt="#,##0", pct_cols=("ошибка, %",))
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)


# ---------------------------------------------------------------- 5. HTML-сводка
def build_summary_html(root: Path, out: Path, meta: dict, checks: dict, diff: pd.DataFrame, oos: pd.DataFrame, nb_res: list):
    """Write the one-page HTML summary: forecast, scenarios, electric share, checks."""
    P, FIG = root / "data/processed", root / "reports/figures"
    tot = pd.read_csv(P / "forecast_4q_total.csv", index_col=0)
    sc = pd.read_csv(P / "long_forecast_totals_layers.csv")
    sh = pd.read_csv(P / "forecast_4q_e_share.csv")

    def f0(v): return f"{v:,.0f}".replace(",", " ")

    def img(name):
        p = FIG / name
        return f'<img src="data:image/png;base64,{base64.b64encode(p.read_bytes()).decode()}" />' if p.exists() else ""

    rows = "".join(f"<tr><td>{n}</td><td>{f0(r['факт: последние 4 кв.'])}</td><td><b>{f0(r['P50'])}</b></td>"
                   f"<td>{f0(r['P10'])} – {f0(r['P90'])}</td><td>{r['P50'] / r['факт: последние 4 кв.'] - 1:+.0%}</td></tr>"
                   for n, r in tot.loc[KEY + [i for i in tot.index if i not in KEY]].iterrows())
    s = sc[(sc["слой"] == "L1") & sc["ряд"].isin(KEY)].pivot_table(index=["ряд", "период"], columns="сценарий", values="P50")
    s3 = sc[(sc["слой"] == "L3") & sc["ряд"].isin(KEY) & (sc["сценарий"] == "Базовый")].set_index(["ряд", "период"])
    srows = "".join(f"<tr><td>{r}</td><td>{p}</td><td>{f0(v['Консервативный'])}</td><td><b>{f0(v['Базовый'])}</b></td>"
                    f"<td>{f0(v['Оптимистичный'])}</td><td>{f0(s3.loc[(r, p), 'P10'])} – {f0(s3.loc[(r, p), 'P90'])}</td></tr>"
                    for (r, p), v in s.reindex(KEY, level=0).iterrows())
    shrows = "".join(f"<tr><td>{r['кв']}</td><td><b>{r['медиана']:.0%}</b></td><td>{r['P10']:.0%} – {r['P90']:.0%}</td></tr>" for _, r in sh.iterrows())
    seen = set()

    def uniq(xs):
        out_ = []
        for x in xs:
            for line in str(x).replace("⚠️", "").strip().splitlines():
                line = line.strip()
                if line and line not in seen:
                    seen.add(line)
                    out_.append(line)
        return out_
    chk = "".join(f'<li class="err">{e}</li>' for e in uniq(checks.get("errors", []))) + \
          "".join(f'<li class="warn">{w}</li>' for w in uniq(checks.get("warnings", []) + checks.get("external", []) +
                                                             [w for r in nb_res for w in r["warnings"]])) + \
          "".join(f"<li>{i}</li>" for i in uniq(checks.get("info", [])))
    nbw = ""
    dtab = ""
    if len(diff):
        dtab = "<h2>Изменение прогноза к прошлому запуску</h2><table><tr><th>ряд</th><th>было (данные до {})</th><th>стало</th><th>изм.</th></tr>".format(diff.iloc[0, 1]) + \
               "".join(f"<tr><td>{r['ряд']}</td><td>{f0(r['прошлый прогноз 4 кв., медиана'])}</td><td>{f0(r['новый прогноз 4 кв., медиана'])}</td>"
                       f"<td>{r['новый прогноз 4 кв., медиана'] / r['прошлый прогноз 4 кв., медиана'] - 1:+.0%}</td></tr>"
                       for _, r in diff[diff['ряд'].isin(KEY)].iterrows()) + "</table>"
    otab = ""
    if len(oos):
        o = oos[oos["ряд"].isin(KEY)]
        otab = "<h2>Как прошлый прогноз попал в новый факт</h2><table><tr><th>квартал</th><th>ряд</th><th>факт</th><th>прогноз</th><th>80 %-ный интервал</th><th>ошибка</th></tr>" + \
               "".join(f"<tr><td>{r['квартал']}</td><td>{r['ряд']}</td><td>{f0(r['факт'])}</td><td>{f0(r['прогноз (медиана)'])}</td>"
                       f"<td>{f0(r['P10'])} – {f0(r['P90'])} {'✓' if r['внутри 80%'] else '✗'}</td><td>{r['ошибка, %']:+.0%}</td></tr>" for _, r in o.iterrows()) + "</table>"
    rule = dvs_rule(pd.read_csv(P / "market_long.csv"))
    rtab = (f"<h2>Решающее правило по ДВС</h2><p>{rule['fact'][0].upper() + rule['fact'][1:]}. "
            f"Вывод: <b>{rule['verdict']}</b> — {rule['action'].rstrip('.')}. Сценарий спроса для плана: «{rule['scenario']}».</p>")
    html = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>СИЛАНТ — сводка прогноза</title>
<style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--mut:#52514e;--line:#e6e5e0;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#1a1a19;--fg:#fff;--mut:#c3c2b7;--line:#3a3a38;--acc:#3987e5}}}}
body{{font-family:Arial,sans-serif;background:var(--bg);color:var(--fg);max-width:1100px;margin:0 auto;padding:16px;line-height:1.45}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid var(--line);padding-bottom:4px}}
table{{border-collapse:collapse;font-size:13px;margin:8px 0;width:100%;display:block;overflow-x:auto}}
td,th{{border-bottom:1px solid var(--line);padding:4px 10px;text-align:right;white-space:nowrap}} td:first-child,th:first-child{{text-align:left}}
th{{color:var(--mut);font-weight:600}} .meta{{color:var(--mut);font-size:13px}}
li.err{{color:#e34948;font-weight:bold}} li.warn{{color:#c98500}} img{{max-width:100%;background:#fff;margin:6px 0}}
</style></head><body>
<h1>СИЛАНТ · сводка прогноза рынка погрузчиков</h1>
<p class="meta">Данные рынка до <b>{meta['last_quarter']}</b> · прогноз {meta['horizon_4q']} (обязательный) и {meta['horizon_8q']} (дополнительный) ·
внешние данные ЦБ на {meta['macro_snapshot']} · запуск {meta['run_time']} · {meta['duration_min']:.1f} мин</p>
<h2>Проверки данных</h2><ul>{chk or '<li>Замечаний нет</li>'}{nbw}</ul>
<h2>Прогноз на 4 квартала ({meta['horizon_4q']}), шт. — модель «Комбинация»</h2>
<table><tr><th>ряд</th><th>факт, последние 4 кв.</th><th>медиана</th><th>80 %-ный интервал</th><th>изм.</th></tr>{rows}</table>
{img('31_fan_market.png')}
<h2>Доля электропогрузчиков</h2><table><tr><th>квартал</th><th>медиана</th><th>80 %-ный интервал</th></tr>{shrows}</table>
<h2>Сценарии (суммы за 4 квартала, медиана), шт.</h2>
<table><tr><th>ряд</th><th>период</th><th>консервативный</th><th>базовый</th><th>оптимистичный</th><th>80 % (база, L3)</th></tr>{srows}</table>
{img('52_fan_long.png')}
{rtab}{dtab}{otab}
<h2>Подробные отчёты</h2><ul>{''.join(f'<li><a href="{n}.html">{n}</a></li>' for n in NOTEBOOKS)}</ul>
<p class="meta">Таблицы: <code>reports/silant_forecast.xlsx</code>, <code>data/processed/*.csv</code>. Текст выводов внутри ноутбуков написан по данным до 2026Q1; цифры на этой странице — всегда актуальные.</p>
</body></html>"""
    out.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------- 6. архив
def archive_run(root: Path, meta: dict, raw_file: Path) -> Path:
    """Copy result tables and run metadata into data/runs/<quarter>__<time>/."""
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    d = root / "data/runs" / f"{meta['last_quarter']}__{ts}"
    d.mkdir(parents=True, exist_ok=True)
    for f in (root / "data/processed").glob("*.csv"):
        shutil.copy2(f, d / f.name)
    for f in (root / "data/processed").glob("*.json"):
        shutil.copy2(f, d / f.name)
    shutil.copy2(raw_file, d / raw_file.name)
    (d / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (root / "data/runs/LATEST.txt").write_text(d.name, encoding="utf-8")
    return d
