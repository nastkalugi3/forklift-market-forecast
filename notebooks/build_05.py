"""Собирает notebooks/05_scenarios.ipynb."""
import nbformat as nbf
from pathlib import Path

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# СИЛАНТ · 05. Сценарии на 8 кварталов

**Три сценария внешних факторов** — базовый (ЦБ), консервативный, оптимистичный; горизонт 8 кварталов (4 обязательных + 4 дополнительных) после последнего факта.

**Модель для сценариев** — та же «Комбинация» (Наивный + ETS + ARX, равные веса) с согласованием и калиброванными интервалами из этапа 04. Отличие одно: ARX теперь двухфакторная — `log-уровень = a + b₁·ставка(t−2) + b₂·инвестиции в ОК г/г(t) + AR(1)`. Коэффициенты сжимаются внутри типа, знаки ограничены априорно (b₁ ≤ 0, b₂ ≥ 0). Наивная и ETS-части от сценария не зависят — это «якорь» к текущему уровню, поэтому модель переносит в прогноз только часть макроэффекта. Отдельно показан «чистый» макроэффект по одной ARX — это верхняя граница чувствительности.

**Почему двухфакторная:** на этапе 03 ставка и инвестиции коррелировали (r = −0,72), и вдвоём по МНК их не разделить. Знаковые ограничения и сжатие к общему по типу коэффициенту работают как информативные априорные оценки. Результат экономически осмысленный: ДВС реагирует на ставку, электро — на инвестиции (см. п. 2).

**Сравнимость:** все сценарии считаются на одних и тех же случайных розыгрышах (общий seed), поэтому различия между ними — только из-за допущений, а не из-за шума симуляции.
""")

code("""
import sys
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mt

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from silant_data import load_market
from silant_ts import build_series
from silant_forecast import forecast_all, rolling_origin, summarize, share_draws, MODELS

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 120)
OUT = ROOT / "data/processed"
FIG = ROOT / "reports/figures"
COL = {"ДВС": "#eb6834", "Электро": "#2a78d6", "Рынок": "#52514e"}
SCOL = {"Консервативный": "#e34948", "Базовый": "#52514e", "Оптимистичный": "#1baf7a"}
SCN = ["Консервативный", "Базовый", "Оптимистичный"]
SCN_SLUG = {"Консервативный": "conservative", "Базовый": "base", "Оптимистичный": "optimistic"}  # ASCII file names
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.6,
        "axes.edgecolor": "#8a8984",
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "lines.linewidth": 2,
    }
)
thous = mt.FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " "))
pctf = mt.PercentFormatter(1.0, decimals=0)


def qlab(p):
    p = pd.Period(p, freq="Q")
    return f"{('I', 'II', 'III', 'IV')[p.quarter - 1]} кв. {p.year}"


def fmt0(v):
    return f"{v:,.0f}".replace(",", " ")


SOURCE_NOTE = (
    "Источники: данные заказчика, Банк России, Росстат, Минэкономразвития России; расчёты автора."
)
UNITS = {"41_scenarios_market": "Продажи, шт.", "43_scenarios_share": "Доля электропогрузчиков"}


def save(fig, name):
    # подпись оси Y (единицы) там, где её нет, и источник данных под рисунком
    unit = UNITS.get(name)
    if unit:
        for ax in fig.axes:
            if not ax.get_ylabel() and ax.get_label() != "<colorbar>":
                ax.set_ylabel(unit)
    fig.canvas.draw()
    bottom = fig.get_tightbbox(fig.canvas.get_renderer()).y0 / fig.get_figheight()
    fig.text(0.01, bottom - 0.02, SOURCE_NOTE, fontsize=8, color="#555555", ha="left", va="top")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.show()


S = build_series(load_market(ROOT / "data/raw/silant_forklift_market.xlsx")[0])
W = pd.read_csv(OUT / "macro_quarterly_wide.csv", index_col=0)
W.index = pd.PeriodIndex(W.index, freq="Q")
INFL = json.loads((OUT / "interval_calibration.json").read_text(encoding="utf-8"))
LAST = S.index.max()
H = 8
FUT = pd.period_range(LAST + 1, periods=H, freq="Q")
Y1, Y2 = FUT[:4], FUT[4:]
KEY = ["Рынок", "ДВС", "Электро"]
print(
    "Последний факт:",
    LAST,
    "| горизонт:",
    qlab(FUT[0]),
    "–",
    qlab(FUT[-1]),
    "| калибровка интервалов:",
    INFL,
)
""")

md("## 1. Допущения по факторам")
code("""
sc_all = pd.read_csv(ROOT / "data/external/manual/scenarios.csv")
from silant_scenarios import scenario_paths

paths, PASS, SC_NOTES = scenario_paths(W, ROOT, LAST, H=H)
for n_ in SC_NOTES:
    print("⚠️", n_)
print("Передача снижения ставки в спрос:", PASS)
kr_fact = W["key_rate_avg"].dropna()
inv_fact = W["invest_fc_yoy"].dropna()
KR_BT = paths["Базовый"]["kr"]  # для бэктеста: факт, где его нет — базовая траектория

need_kr = pd.period_range(FUT[0] - 2, FUT[-1] - 2, freq="Q")
need_inv = FUT
tab = []
for s in SCN:
    tab.append(
        pd.Series(
            paths[s]["kr"].reindex(need_kr).to_numpy(),
            index=[f"ставка {qlab(q)}" for q in need_kr],
            name=s,
        )
    )
kr_tab = pd.concat(tab, axis=1)
inv_tab = pd.concat(
    [
        pd.Series(
            paths[s]["inv"].reindex(need_inv).to_numpy(),
            index=[f"инвестиции {qlab(q)}" for q in need_inv],
            name=s,
        )
        for s in SCN
    ],
    axis=1,
)
factual = [f"ставка {qlab(q)}" for q in need_kr if q in kr_fact.index] + [
    f"инвестиции {qlab(q)}" for q in need_inv if q in inv_fact.index
]
print("Фактические значения (одинаковы во всех сценариях):", factual)
display(pd.concat([kr_tab, inv_tab]).round(2))

ann = []
for s in SCN:
    k, i = paths[s]["kr"], paths[s]["inv"]
    ann.append(
        {
            "сценарий": s,
            "ставка 2026, ср.": k[k.index.year == 2026].mean(),
            "ставка 2027, ср.": k[k.index.year == 2027].mean(),
            "инвестиции II пол. 2026, г/г": i[
                (i.index.year == 2026) & (i.index.quarter > 2)
            ].mean(),
            "инвестиции 2027, г/г": i[i.index.year == 2027].mean(),
        }
    )
ann = pd.DataFrame(ann).set_index("сценарий").round(2)
display(ann)
sc_all.to_csv(OUT / "scenario_inputs.csv", index=False, encoding="utf-8-sig")
""")

md("""
| Фактор | Консервативный | Базовый (ЦБ) | Оптимистичный | Источник допущения |
|---|---|---|---|---|
| Ключевая ставка, ср. 2026 | 14,6 % (пауза на 14 % до конца года) | 14,5 % | 14,5 % (снижение в октябре) | ЦБ, 24.07.2026: 14,5–14,6 %; фактическая ставка I–III кв. 2026 одинакова во всех сценариях |
| Ключевая ставка, ср. 2027 | 12,5 % (верхняя граница ЦБ; ≈ медиана опроса аналитиков 12,4 %) | 11,5 % (середина 10,5–12,5 %) | 10,5 % (нижняя граница ЦБ) | ЦБ, среднесрочный прогноз; опрос ЦБ, сентябрь 2026 |
| Инвестиции в основной капитал, II пол. 2026 | ≈ −6 % г/г (год ≈ −7,5 %) | ≈ −2,4 % (год −5,4 %) | ≈ 0 % (год ≈ −4 %) | Минэкономразвития, сент. 2026: −5,4 % за 2026 при факте I пол. −9,9 %; консервативный и оптимистичный варианты — стресс-допущения |
| Инвестиции в основной капитал, 2027 | −2 % | +0,2 % | +3 % | Минэкономразвития +0,2 %; ЦБ (валовое накопление ОК) +2…+4 % |
| **Передача снижения ставки в спрос** | **50 %** | 100 % | 100 % | Допущение о механизме, а не о факторе: в данных один цикл ставки, и неизвестно, вернётся ли спрос на ДВС так же быстро, как упал. В I кв. 2026 восстановления ещё не было, хотя ставка с лагом уже снизилась на 2,6 п. п. |
| Курс USD/RUB | в модель не входит | в модель не входит | в модель не входит | на этапе 03 знак влияния не подтвердился; справочно: опрос ЦБ 88,8 (2027), Минэкономразвития 87,4 |

Для прогноза продаж в квартале t нужна ставка за квартал t−2. Поэтому на **II кв. 2026 — III кв. 2026 все сценарии совпадают по ставке** (используется фактическая ставка IV кв. 2025 — I кв. 2026), а инвестиции II кв. 2026 — тоже факт (−6,6 %). Сценарии начинают расходиться с III кв. 2026 по инвестициям и с IV кв. 2026 по ставке.
""")

code("""
fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
for s in SCN:
    k = paths[s]["kr"][paths[s]["kr"].index >= pd.Period("2024Q1", freq="Q")]
    i = paths[s]["inv"][paths[s]["inv"].index >= pd.Period("2024Q1", freq="Q")]
    axes[0].plot(k.index.to_timestamp(), k, color=SCOL[s], marker="o", ms=3.5, label=s)
    axes[1].plot(i.index.to_timestamp(), i, color=SCOL[s], marker="o", ms=3.5, label=s)
for ax, t in zip(
    axes, ["Ключевая ставка, % (ср. за квартал)", "Инвестиции в основной капитал, % г/г"]
):
    ax.axvspan(
        FUT[0].to_timestamp() - pd.Timedelta(days=45),
        FUT[-1].to_timestamp(),
        color="#8a8984",
        alpha=0.08,
        lw=0,
    )
    ax.set_title(t)
    qs = pd.period_range("2024Q1", FUT[-1], freq="Q")[::2]
    ax.set_xticks(qs.to_timestamp(), [qlab(q) for q in qs], rotation=45)
axes[1].axhline(0, color="#8a8984", lw=0.8)
axes[0].legend(frameon=False)
fig.suptitle(
    "Допущения сценариев (серым — горизонт прогноза; до расхождения линий — факт)",
    fontweight="bold",
)
fig.tight_layout()
save(fig, "40_scenario_inputs")
""")

md("## 2. Эластичности модели (что именно двигает прогноз)")
code("""
res, pure = {}, {}
for s in SCN:
    res[s] = forecast_all(
        S,
        paths[s]["kr"],
        LAST,
        H=H,
        n_sim=4000,
        seed=SEED,
        infl=INFL,
        inv=paths[s]["inv"],
        kr_pass=PASS[s],
    )
fc0 = res["Базовый"]
el = pd.DataFrame(
    {
        n: {
            "+1 п.п. ставки (t−2)": np.expm1(v["b"][0]),
            "+1 п.п. роста инвестиций (t)": np.expm1(v["b"][1]),
            "φ": v["phi"],
        }
        for n, v in fc0["arx"].items()
    }
).T
order = (
    KEY
    + [b for b in fc0["bottom"] if b.startswith("ДВС")]
    + [b for b in fc0["bottom"] if b.startswith("Электро")]
)
el.loc[order].style.format(
    {"+1 п.п. ставки (t−2)": "{:+.1%}", "+1 п.п. роста инвестиций (t)": "{:+.1%}", "φ": "{:.2f}"}
)
""")

code("""
# проверка двухфакторной ARX на rolling-origin (условный прогноз: фактические значения факторов известны)
inv_bt = inv_fact.reindex(pd.period_range(inv_fact.index.min(), LAST + 4, freq="Q")).ffill()
bt1 = rolling_origin(S, KR_BT, pd.Period("2024Q4", freq="Q"), LAST - 1, H=4, n_sim=500, seed=SEED)
rows = []
for k, o in enumerate(pd.period_range("2024Q4", LAST - 1, freq="Q")):
    f_ = forecast_all(S, KR_BT, o, H=4, n_sim=500, seed=SEED + k, inv=inv_bt)
    for m in ["ARX-ставка", "Комбинация"]:
        for n in KEY:
            for h, q in enumerate(f_["fut"]):
                if q in S.index:
                    rows.append(
                        {
                            "model": m,
                            "node": n,
                            "err": abs(np.log(S.loc[q, n] / f_["rec"][m][n][:, h].mean())),
                        }
                    )
two = pd.DataFrame(rows).pivot_table(index="model", columns="node", values="err")[KEY]
one = bt1[
    (bt1.src == "rec") & bt1.node.isin(KEY) & bt1.model.isin(["ARX-ставка", "Комбинация"])
].pivot_table(index="model", columns="node", values="abs_log_err")[KEY]
cmp_ = pd.concat({"только ставка": one, "ставка + инвестиции": two}).swaplevel().sort_index()
cmp_.style.format("{:.1%}").set_caption(
    "Средняя ошибка (rolling-origin, h=1–4). Для двухфакторной модели фактические инвестиции в будущих кварталах считались известными — это оценка условного прогноза, она оптимистична"
)
""")

md("## 3. Прогноз по сценариям")
code("""
def tot(sims, n, qs_):
    idx = [list(FUT).index(q) for q in qs_]
    return sims[n][:, idx].sum(1)


rows = []
for s in SCN:
    sims = res[s]["rec"]["Комбинация"]
    for n in res[s]["nodes"]:
        for per, qs_ in [("Y1", Y1), ("Y2", Y2)]:
            v = tot(sims, n, qs_)
            rows.append(
                {
                    "сценарий": s,
                    "ряд": n,
                    "период": per,
                    "P10": np.quantile(v, 0.1),
                    "P50": np.median(v),
                    "P90": np.quantile(v, 0.9),
                    "P2.5": np.quantile(v, 0.025),
                    "P97.5": np.quantile(v, 0.975),
                    "mean": v.mean(),
                }
            )
    summarize(sims, FUT).assign(scenario=s).to_csv(
        OUT / f"scenario_forecast_q_{SCN_SLUG[s]}.csv", index=False, encoding="utf-8-sig"
    )
R = pd.DataFrame(rows)
R.to_csv(OUT / "scenario_forecast_totals.csv", index=False, encoding="utf-8-sig")
base_last4 = S.iloc[-4:].sum()
P1, P2 = f"{qlab(Y1[0])}–{qlab(Y1[-1])}", f"{qlab(Y2[0])}–{qlab(Y2[-1])}"
wide = R.pivot_table(index="ряд", columns=["период", "сценарий"], values="P50").reindex(order)
out = pd.DataFrame(index=order)
out["факт: последние 4 кв."] = base_last4.reindex(order)
for per, lab in [("Y1", P1), ("Y2", P2)]:
    for s in SCN:
        out[f"{lab} · {s}"] = wide[(per, s)]
out.style.format(fmt0).set_caption("Медиана продаж за 4 квартала, шт.")
""")

code("""
chg = pd.DataFrame(index=order)
for per, lab in [("Y1", P1), ("Y2", P2)]:
    for s in ["Консервативный", "Оптимистичный"]:
        chg[f"{lab} · {s} к базовому"] = wide[(per, s)] / wide[(per, "Базовый")] - 1
chg[f"{P2} · базовый к последним 4 кв."] = wide[("Y2", "Базовый")] / base_last4.reindex(order) - 1
chg.style.format("{:+.0%}").set_caption("Насколько сценарий отличается от базового (по медиане)")
""")

code("""
ci = R[R["ряд"].isin(KEY)].copy()
ci["80 %-ный интервал"] = [f"{fmt0(a)} – {fmt0(b)}" for a, b in zip(ci.P10, ci.P90)]
ci["медиана"] = ci.P50.map(fmt0)
ci["период"] = ci["период"].map({"Y1": P1, "Y2": P2})
ci.pivot_table(
    index=["ряд", "сценарий"],
    columns="период",
    values=["медиана", "80 %-ный интервал"],
    aggfunc="first",
).reindex(KEY, level=0)
""")

code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
hist = S.index.to_timestamp()
for ax, n in zip(axes, KEY):
    ax.plot(hist, S[n], color=COL[n], marker="o", ms=3.5, label="факт")
    b = summarize(res["Базовый"]["rec"]["Комбинация"], FUT)
    b = b[b.node == n]
    x = FUT.to_timestamp()
    ax.fill_between(x, b.q100, b.q900, color="#8a8984", alpha=0.18, lw=0, label="80 % (базовый)")
    for s in SCN:
        d = summarize(res[s]["rec"]["Комбинация"], FUT)
        d = d[d.node == n]
        ax.plot(
            [hist[-1]] + list(x),
            [S[n].iloc[-1]] + list(d.q500),
            color=SCOL[s],
            lw=2 if s == "Базовый" else 1.6,
            ls="-" if s == "Базовый" else "--",
            marker="o",
            ms=3,
            label=s,
        )
    ax.set_title(n)
    ax.yaxis.set_major_formatter(thous)
    ax.set_ylim(bottom=0)
    t = list(S.index[::2]) + list(FUT[1::2])
    ax.set_xticks([q.to_timestamp() for q in t], [qlab(q) for q in t], rotation=45)
axes[0].legend(frameon=False, fontsize=8, loc="upper right")
fig.suptitle(
    "Сценарии: медиана по кварталам и 80 %-ный интервал базового сценария", fontweight="bold"
)
fig.tight_layout()
save(fig, "41_scenarios_market")
""")

code("""
segs = [n for n in order if n not in KEY]
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
for ax, per, lab in zip(axes, ["Y1", "Y2"], [P1, P2]):
    for i, n in enumerate(segs[::-1]):
        for j, s in enumerate(SCN):
            r = R[(R["ряд"] == n) & (R["период"] == per) & (R["сценарий"] == s)].iloc[0]
            y = i + (j - 1) * 0.22
            ax.hlines(y, r.P10, r.P90, color=SCOL[s], lw=3, alpha=0.45)
            ax.plot(
                r.P50,
                y,
                "o",
                color=SCOL[s],
                ms=5.5,
                mec="white",
                mew=1,
                label=s if i == 0 else None,
            )
        ax.plot(
            base_last4[n],
            i,
            "D",
            color="#0b0b0b",
            mfc="white",
            ms=5,
            label="последние 4 кв. факта" if i == 0 else None,
        )
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(thous)
    ax.set_title(f"{lab}: медиана и 80 %-ный интервал, шт.")
    ax.set_yticks(range(len(segs)), segs[::-1])
axes[0].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout()
save(fig, "42_scenarios_segments")
""")

md("## 4. Доля электропогрузчиков по сценариям")
code("""
rows = []
for s in SCN:
    sims = res[s]["rec"]["Комбинация"]
    for per, qs_ in [("Y1", Y1), ("Y2", Y2)]:
        idx = [list(FUT).index(q) for q in qs_]
        v = sims["Электро"][:, idx].sum(1) / sims["Рынок"][:, idx].sum(1)
        row = {
            "сценарий": s,
            "период": P1 if per == "Y1" else P2,
            "медиана": np.median(v),
            "P10": np.quantile(v, 0.1),
            "P90": np.quantile(v, 0.9),
        }
        for cls in ["1–1,8 т", "2–2,5 т", "3–3,5 т", "3,8–5 т"]:
            e_, d_ = sims[f"Электро {cls}"][:, idx].sum(1), sims[f"ДВС {cls}"][:, idx].sum(1)
            row[f"класс {cls}"] = np.median(e_ / (e_ + d_))
        rows.append(row)
SH = pd.DataFrame(rows)
SH.to_csv(OUT / "scenario_e_share.csv", index=False, encoding="utf-8-sig")
print(f"Факт за последние 4 кв.: {S['Электро'].iloc[-4:].sum() / S['Рынок'].iloc[-4:].sum():.1%}")
SH.set_index(["период", "сценарий"]).style.format("{:.0%}")
""")

code("""
fig, ax = plt.subplots(figsize=(9, 3.6))
sh = S["Электро"] / S["Рынок"]
ax.plot(S.index.to_timestamp(), sh, color=COL["Электро"], marker="o", ms=4, label="факт")
for s in SCN:
    sims = res[s]["rec"]["Комбинация"]
    v = share_draws(sims)
    ax.plot(
        [S.index[-1].to_timestamp()] + list(FUT.to_timestamp()),
        [sh.iloc[-1]] + list(np.median(v, 0)),
        color=SCOL[s],
        ls="--" if s != "Базовый" else "-",
        marker="o",
        ms=3,
        label=s,
    )
v = share_draws(res["Базовый"]["rec"]["Комбинация"])
ax.fill_between(
    FUT.to_timestamp(),
    np.quantile(v, 0.1, 0),
    np.quantile(v, 0.9, 0),
    color="#8a8984",
    alpha=0.18,
    lw=0,
    label="80 % (базовый)",
)
ax.yaxis.set_major_formatter(pctf)
ax.set_ylim(0.35, 0.95)
t = list(S.index[::2]) + list(FUT[1::2])
ax.set_xticks([q.to_timestamp() for q in t], [qlab(q) for q in t], rotation=45)
ax.set_title("Доля электропогрузчиков по сценариям (медиана)")
ax.legend(frameon=False, fontsize=8, ncol=2)
save(fig, "43_scenarios_share")
""")

md("""
## 5. Из чего складывается прогноз: вклад компонентов
В «Комбинации» от сценария зависит только ARX-часть (1/3 веса в логарифме). Наивная и ETS-части говорят: «уровень останется, как в последнем квартале, с поправкой на сезон». ARX говорит: «рынок вернётся туда, где он был при такой же ставке и инвестициях». Ниже медианы каждой части по сценариям. Это показывает, что прогноз на самом деле держится не столько на выборе сценария, сколько на выборе между «уровень сохранится» и «рынок вернётся к докризисному уровню при той же ставке».
""")
code("""
rows = []
for s in SCN:
    for n in KEY:
        for per, qs_ in [("Y1", Y1), ("Y2", Y2)]:
            idx = [list(FUT).index(q) for q in qs_]
            r_ = {"ряд": n, "период": P1 if per == "Y1" else P2, "сценарий": s}
            for m in ["Наивный", "ETS", "ARX-ставка", "Комбинация"]:
                r_[m] = np.median(res[s]["rec"][m][n][:, idx].sum(1))
            rows.append(r_)
M = pd.DataFrame(rows).rename(columns={"ARX-ставка": "ARX (ставка + инвестиции)"})
M.to_csv(OUT / "scenario_components.csv", index=False, encoding="utf-8-sig")
M["факт: последние 4 кв."] = M["ряд"].map(base_last4)
M.set_index(["ряд", "период", "сценарий"]).reindex(KEY, level=0).style.format(fmt0)
""")
code("""
spread = M.pivot_table(
    index=["ряд", "период"], columns="сценарий", values=["Комбинация", "ARX (ставка + инвестиции)"]
)
rng_ = pd.DataFrame(
    {
        m: spread[(m, "Оптимистичный")] / spread[(m, "Консервативный")] - 1
        for m in ["Комбинация", "ARX (ставка + инвестиции)"]
    }
)
rng_.columns = ["разрыв опт./конс.: Комбинация", "разрыв опт./конс.: только ARX (верхняя граница)"]
rng_.reindex(KEY, level=0).style.format("{:+.0%}")
""")

md("""
### 5.1 Проверка устойчивости: сценарии на реальной ставке
Те же сценарии, но в модели ARX вместо номинальной ключевой ставки — реальная (ставка минус ИПЦ г/г; на будущее — траектория инфляции по прогнозу Банка России: 6,5 % в IV кв. 2026 г. → 4,0 % в IV кв. 2027 г., `data/external/manual/cpi_forecast.csv`). Инфляция одинакова во всех сценариях, различаются номинальная ставка и инвестиции.
""")
code("""
from silant_scenarios import real_rate

Y1i, Y2i = slice(0, 4), slice(4, 8)
rows = []
for s in SCN:
    for name, kr_ in [
        ("номинальная", paths[s]["kr"]),
        ("реальная", real_rate(paths[s]["kr"], ROOT)),
    ]:
        f_ = forecast_all(
            S,
            kr_,
            LAST,
            H=H,
            n_sim=3000,
            seed=SEED,
            infl=INFL,
            inv=paths[s]["inv"],
            kr_pass=PASS[s],
        )
        for m in ["Комбинация", "ARX-ставка"]:
            for n in ["Рынок", "ДВС"]:
                sims = f_["rec"][m][n]
                rows.append(
                    {
                        "сценарий": s,
                        "ставка": name,
                        "модель": m,
                        "ряд": n,
                        "Y1": float(np.median(sims[:, Y1i].sum(1))),
                        "Y2": float(np.median(sims[:, Y2i].sum(1))),
                    }
                )
RRS = pd.DataFrame(rows)
RRS.to_csv(OUT / "scenario_real_rate_check.csv", index=False, encoding="utf-8-sig")
RRS.pivot_table(index=["модель", "ряд", "ставка"], columns="сценарий", values=["Y1", "Y2"])[
    [("Y1", s) for s in SCN] + [("Y2", s) for s in SCN]
].style.format(lambda v: f"{v / 1000:.1f}").set_caption(
    "Медиана продаж за 4 квартала, тыс. шт.: Y1 = II кв. 2026 — I кв. 2027, Y2 = II кв. 2027 — I кв. 2028"
)
""")

md("""
### 5.2 Сценарии реакции спроса
Макросценарии (раздел 3) различаются на 2–4 %: ставка на первый год уже известна из-за лага, а коридор официальных прогнозов узкий. Для завода важнее другие развилки — **как спрос отреагирует** на уже известные условия. Строим связные сценарии с прозрачными центральными траекториями (сезонно скорректированный уровень × общий сезонный профиль или медиана модели по кварталам) и одинаковой относительной шириной интервала, взятой из основной модели:

1. **Спад сохраняется** — продажи всех сегментов остаются на сезонно скорректированном уровне I кв. 2026 г.;
2. **Ускоренное замещение** — рынок на том же уровне, доля электро продолжает расти (75 % в первый год, 78 % во второй; в 2023–2025 гг. доля росла на 5–11 п. п. в год);
3. **I кв. 2026 занижен покупками на опережение** — стартовый уровень = среднее сезонно скорректированных IV кв. 2025 и I кв. 2026 г.;
4. **Восстановление ДВС вслед за ставкой** — ДВС по модели ARX (базовые ставка и инвестиции), электро на текущем уровне;
5. **Базовый макросценарий** — комбинация моделей из раздела 3 (для сравнения).

Центральная оценка здесь — **сумма поквартальных медиан**: она не содержит статистической асимметрии суммы, поэтому сценарии сравниваются по существу.
""")
code("""
from silant_forecast import season_profile

prof = np.exp(season_profile(S.loc[:LAST, "Электро"])[0])
fq = np.array([q.quarter - 1 for q in FUT])
fb = res["Базовый"]
bottom = fb["bottom"]
dvs_b = [b for b in bottom if b.startswith("ДВС")]
el_b = [b for b in bottom if b.startswith("Электро")]
lvl_now = {b: S.loc[LAST, b] / prof[LAST.quarter - 1] for b in bottom}
lvl_pf = {
    b: np.sqrt(lvl_now[b] * S.loc[LAST - 1, b] / prof[(LAST - 1).quarter - 1]) for b in bottom
}
flat = lambda lvl: lvl * prof[fq]
med = lambda m, b: np.median(fb["rec"][m][b], axis=0)
DS = {
    "Спад сохраняется": {b: flat(lvl_now[b]) for b in bottom},
    "I кв. 2026 занижен (опережение)": {b: flat(lvl_pf[b]) for b in bottom},
    "Восстановление ДВС вслед за ставкой": {
        **{b: med("ARX-ставка", b) for b in dvs_b},
        **{b: flat(lvl_now[b]) for b in el_b},
    },
    "Базовый макросценарий (комбинация)": {b: med("Комбинация", b) for b in bottom},
}
SHARE_T = {"Y1": 0.75, "Y2": 0.78}
# относительная ширина 80 % интервала годовой суммы из основной модели (базовый макросценарий)
comb = fb["rec"]["Комбинация"]


def rel_width(n, sl):
    v = comb[n][:, sl].sum(1)
    return np.quantile(v, 0.1) / np.median(v), np.quantile(v, 0.9) / np.median(v)


SEG = [b for b in bottom if "5,5" not in b]  # 8 крупных сегментов (линейка СИЛАНТ — 1,5–5 т)
rows = []
for per, sl in [("Y1", slice(0, 4)), ("Y2", slice(4, 8))]:
    flat_mkt = sum(DS["Спад сохраняется"][b][sl].sum() for b in bottom)
    scen = {}
    for s, pth in DS.items():
        d = sum(pth[b][sl].sum() for b in dvs_b)
        e = sum(pth[b][sl].sum() for b in el_b)
        scen[s] = {"ДВС": d, "Электро": e, **{g: pth[g][sl].sum() for g in SEG}}
    fd = scen["Спад сохраняется"]
    sub = {"ДВС": flat_mkt * (1 - SHARE_T[per]), "Электро": flat_mkt * SHARE_T[per]}
    # сегменты: структура внутри типа как в сценарии «спад сохраняется», масштаб — по итогу типа
    sub.update({g: fd[g] * sub[g.split(" ")[0]] / fd[g.split(" ")[0]] for g in SEG})
    scen["Ускоренное замещение"] = sub
    for s, v in scen.items():
        mkt = v["ДВС"] + v["Электро"]
        r = {
            "сценарий": s,
            "период": per,
            "Рынок": mkt,
            "ДВС": v["ДВС"],
            "Электро": v["Электро"],
            "доля электро": v["Электро"] / mkt,
        }
        for n in KEY:
            lo, hi = rel_width(n, sl)
            r[f"{n} P10"] = r[n] * lo
            r[f"{n} P90"] = r[n] * hi
        r.update({g: v.get(g, np.nan) for g in SEG})
        r["ДВС в 1-м прогнозном квартале"] = sum(DS[s][b][0] for b in dvs_b) if s in DS else np.nan
        rows.append(r)
DEM = pd.DataFrame(rows)
DEM.to_csv(OUT / "scenario_demand.csv", index=False, encoding="utf-8-sig")
order_s = [
    "Спад сохраняется",
    "Ускоренное замещение",
    "I кв. 2026 занижен (опережение)",
    "Базовый макросценарий (комбинация)",
    "Восстановление ДВС вслед за ставкой",
]
show = (
    DEM.set_index(["сценарий", "период"])[
        ["Рынок", "ДВС", "Электро", "доля электро"]
        + [
            "ДВС 3–3,5 т",
            "Электро 1–1,8 т",
            "Электро 2–2,5 т",
            "Электро 3–3,5 т",
            "Электро 3,8–5 т",
        ]
    ]
    .unstack("период")
    .reindex(order_s)
)
show.style.format(
    lambda v: "—" if pd.isna(v) else (f"{v:.0%}" if v < 1.5 else fmt0(v))
).set_caption(f"Сценарии реакции спроса: сумма поквартальных медиан, шт. (Y1 = {P1}, Y2 = {P2})")
""")

md("""
Тот же результат на графике: центральные значения сценариев и 80 %-ный интервал основной модели для рынка, ДВС и электро.
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
cols = ["#8a8984", "#2a78d6", "#eda100", "#52514e", "#eb6834"]
for ax, n in zip(axes, KEY):
    for k, s in enumerate(order_s):
        for j, per in enumerate(["Y1", "Y2"]):
            r = DEM[(DEM["сценарий"] == s) & (DEM["период"] == per)].iloc[0]
            x = j + (k - 2) * 0.13
            ax.plot(
                [x, x], [r[f"{n} P10"] / 1e3, r[f"{n} P90"] / 1e3], color=cols[k], lw=1.2, alpha=0.5
            )
            ax.plot(
                x,
                r[n] / 1e3,
                "o",
                color=cols[k],
                ms=7,
                label=s if (n == "Рынок" and j == 0) else None,
            )
    ax.axhline(S[n].iloc[-4:].sum() / 1e3, color="#0b0b0b", lw=1, ls="--")
    ax.set_xticks([0, 1], [P1, P2])
    ax.set_title(f"{n}, тыс. шт. за 4 кв.")
axes[0].legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Сценарии реакции спроса (пунктир — факт последних 4 кв.)", fontweight="bold")
fig.tight_layout()
save(fig, "44_demand_scenarios")
""")

md("""
### 5.3 Обратный стресс-тест: что должно произойти, чтобы спрос на ДВС восстановился
Вместо вопроса «что будет при ставке X» ставим обратный: при каких условиях модель ожидает возврата продаж ДВС, и объясняют ли ставка и инвестиции нынешний уровень продаж. Используется модель ARX для ДВС в целом: сезонно скорректированный уровень = exp(a + b·ставка(t−2) + c·инвестиции(t)).
""")
code("""
a_ = fb["arx"]["ДВС"]
nm = a_["names"]
bk, bi = a_["b"][nm.index("kr")], a_["b"][nm.index("inv")]
krp, invp = paths["Базовый"]["kr"], paths["Базовый"]["inv"]
idx = S.index
hist = pd.DataFrame(
    {
        "ставка t−2, %": [krp.get(q - 2, np.nan) for q in idx],
        "инвестиции г/г, %": [invp.get(q, np.nan) for q in idx],
        "ДВС, факт": S["ДВС"].to_numpy(),
        "ДВС без сезонности": S["ДВС"].to_numpy() / prof[idx.quarter - 1],
    },
    index=idx,
)
hist["уровень по ARX"] = np.exp(
    a_["a"] + bk * hist["ставка t−2, %"] + bi * hist["инвестиции г/г, %"]
)
hist["факт к ARX"] = hist["ДВС без сезонности"] / hist["уровень по ARX"] - 1
fwd = pd.DataFrame(
    {
        "ставка t−2, %": [krp[q - 2] for q in FUT[:4]],
        "инвестиции г/г, %": [invp[q] for q in FUT[:4]],
    },
    index=FUT[:4],
)
fwd["уровень по ARX"] = np.exp(a_["a"] + bk * fwd["ставка t−2, %"] + bi * fwd["инвестиции г/г, %"])
fwd["ДВС: спад сохраняется"] = [sum(DS["Спад сохраняется"][b][i] for b in dvs_b) for i in range(4)]
fwd["ДВС: восстановление (ARX)"] = [
    sum(DS["Восстановление ДВС вслед за ставкой"][b][i] for b in dvs_b) for i in range(4)
]
sa_last = hist["ДВС без сезонности"].iloc[-1]
r_need = (np.log(sa_last) - a_["a"] - bi * invp[LAST]) / bk
RS = pd.concat([hist.assign(часть="история"), fwd.assign(часть="прогноз")])
RS.index = RS.index.astype(str)
RS.to_csv(OUT / "scenario_reverse_stress.csv", encoding="utf-8-sig")
print(
    f"Эффект: +1 п.п. ставки (t−2) → {np.expm1(bk):+.1%} к уровню ДВС; +1 п.п. инвестиций → {np.expm1(bi):+.1%}"
)
print(
    f"Чтобы объяснить нынешний уровень ДВС ({fmt0(sa_last)} шт./кв. без сезонности) ставкой при нынешних инвестициях, она должна была бы быть {r_need:.1f} % (максимум в истории — 21 %)"
)
for tgt in (7500, 8000):
    r_t = (np.log(tgt) - a_["a"] - bi * 0.0) / bk
    print(
        f"Модель ARX ожидает ДВС ≥ {fmt0(tgt)} шт./кв. (без сезонности, инвестиции 0 % г/г) при ставке t−2 не выше {r_t:.1f} %"
    )
display(
    hist.round(2)
    .rename(index=qlab)
    .style.format("{:.1f}", subset=["ставка t−2, %", "инвестиции г/г, %"])
    .format(fmt0, subset=["ДВС, факт", "ДВС без сезонности", "уровень по ARX"])
    .format("{:+.0%}", subset=["факт к ARX"])
)
fwd.rename(index=qlab).style.format("{:.1f}", subset=["ставка t−2, %", "инвестиции г/г, %"]).format(
    fmt0, subset=["уровень по ARX", "ДВС: спад сохраняется", "ДВС: восстановление (ARX)"]
)
""")

md("""
## 6. Выводы

### 6.1 Что двигает прогноз в модели
- **ДВС реагирует на ставку, электро — на инвестиции.** +1 п. п. ключевой ставки (с лагом 2 кв.) → **−4,7 %** продаж ДВС. Для электро эффект ставки нулевой (оценка получилась положительной, но априорное ограничение знака её обнулило). +1 п. п. роста инвестиций в основной капитал → **+1,7 %** продаж электро и +0,3 % ДВС. По рынку в целом: −1,6 % на 1 п. п. ставки и +1,2 % на 1 п. п. инвестиций.
- **Две переменные вместо одной улучшают условный прогноз.** На rolling-origin ошибка комбинации по рынку снизилась с 29 % до 22 %, по электро — с 20 % до 11 %; по ДВС почти без изменений (50 %). Это верхняя оценка качества: в проверке будущие инвестиции брались фактическими, а в реальном прогнозе они тоже неизвестны.

### 6.2 Сценарии: прогноз по сегментам (медиана за 4 квартала, шт.)
| | Факт, последние 4 кв. | II кв. 2026 — I кв. 2027: конс. / база / опт. | II кв. 2027 — I кв. 2028: конс. / база / опт. |
|---|---|---|---|
| **Рынок** | 76 000 | 85 500 / **87 200** / 87 800 | 92 800 / **95 200** / 96 800 |
| ДВС | 21 600 | 29 100 / **29 800** / 29 900 | 30 800 / **32 600** / 33 100 |
| Электро | 54 400 | 55 600 / **56 400** / 57 000 | 57 700 / **58 300** / 59 100 |
| ДВС 3–3,5 т | 8 400 | 11 300 / 11 600 / 11 600 | 11 300 / 12 000 / 12 100 |
| ДВС 1–1,8 т | 5 900 | 7 300 / 7 500 / 7 500 | 7 100 / 7 600 / 7 600 |
| ДВС 2–2,5 т | 4 800 | 6 600 / 6 800 / 6 800 | 6 700 / 7 000 / 7 200 |
| Электро 1–1,8 т | 24 600 | 25 500 / 25 800 / 26 100 | 25 100 / 25 400 / 25 800 |
| Электро 2–2,5 т | 14 300 | 16 000 / 16 200 / 16 400 | 15 800 / 16 000 / 16 200 |
| Электро 3–3,5 т | 12 800 | 10 600 / 10 700 / 10 900 | 10 300 / 10 400 / 10 600 |
| Электро 3,8–5 т | 2 400 | 2 300 / 2 300 / 2 300 | 2 200 / 2 200 / 2 200 |
| Доля электро | 71,6 % | 66 % / 65 % / 66 % | 66 % / 65 % / 65 % |

80 %-ный интервал базового сценария по рынку: 63,5–116,9 тыс. (II кв. 2026 — I кв. 2027) и 58,2–148,4 тыс. (II кв. 2027 — I кв. 2028).

⚠️ Рост второго года к первому в том числе статистический: медиана суммы кварталов растёт вместе с шириной распределения. Даже наивная модель «без изменений» даёт +11 % по рынку. Корректно сравнивать сценарии между собой внутри одного периода.

### 6.3 Главный вывод: выбор сценария важен меньше, чем кажется
1. **Разница между сценариями — единицы процентов:** консервативный сценарий ниже базового на 2–5 %, оптимистичный выше на 1–2 %. Интервал неопределённости модели шире в разы (±35–55 %). Причины три:
   - на первый год ставка уже известна (лаг 2 квартала);
   - опубликованные прогнозы ЦБ расходятся всего на ±1 п. п. ставки в 2027 году;
   - в комбинацию макроэффект входит с весом 1/3.
2. **Даже если рынок полностью следует за факторами (одна ARX-модель),** разрыв между оптимистичным и консервативным сценарием во втором году — около 14 % по рынку, 24 % по ДВС и 8 % по электро. Это верхняя граница чувствительности.
3. **Настоящая развилка — вернётся ли ДВС.** Если «уровень сохранится» (наивная модель и ETS), рынок ДВС составит 26–28 тыс. шт. в год. Если «рынок вернётся туда, где был при такой ставке» (ARX), — 34–46 тыс. Разница около 1,5 раза, и она больше, чем между любыми сценариями по факторам. Консервативное допущение о 50 %-й передаче снижения ставки в спрос сдвигает ДВС вниз всего на 3–5 %, потому что вес ARX в комбинации — треть.
4. **Доля электро во всех сценариях ~65–66 %,** по классам: 1–1,8 т ≈ 77–78 %, 2–2,5 т ≈ 70 %, 3–3,5 т ≈ 48 %, 3,8–5 т ≈ 57–61 %. Центральное снижение с 72 % — следствие ожидаемого частичного восстановления ДВС. Если восстановления не будет, доля останется на уровне ~70–72 %.

5. **Реальная ставка (раздел 5.1).** Если в модели ARX заменить номинальную ставку на реальную (за вычетом инфляции), прогноз комбинации почти не меняется (рынок во втором году ~94 тыс. вместо ~96 тыс. в базовом сценарии). Но верхняя граница «возврата ДВС» по ARX снижается с 34–46 до 32–40 тыс. шт. в год: часть снижения номинальной ставки — это замедление инфляции, а не удешевление денег. Восстановление ДВС до уровня 2024 г. при реальной ставке 6–9 % (в 2023 г. — около 4 %) маловероятно.

### 6.3а Сценарии реакции спроса и обратный стресс-тест (разделы 5.2–5.3)
1. **Развилки спроса дают разброс, который имеет значение для плана** (сумма поквартальных медиан, первый / второй год):

| Сценарий | Рынок, тыс. шт. | ДВС, тыс. шт. | Электро, тыс. шт. | Доля электро |
|---|---|---|---|---|
| Спад сохраняется | 77,3 / 77,3 | 23,8 / 23,8 | 53,4 / 53,4 | 69 % |
| Ускоренное замещение | 77,3 / 77,3 | 19,3 / 17,0 | 58,0 / 60,3 | 75 / 78 % |
| I кв. 2026 занижен (опережение) | 76,5 / 76,5 | 22,1 / 22,1 | 54,4 / 54,4 | 71 % |
| Базовый макросценарий (комбинация) | 81,5 / 80,5 | 27,4 / 27,6 | 54,1 / 52,9 | 66 % |
| Восстановление ДВС вслед за ставкой | 88,1 / 94,0 | 34,6 / 40,6 | 53,4 / 53,4 | 61 / 57 % |

   Рынок различается на 15 % в первый год и на 23 % во второй, но главное — **ДВС различается в 1,8 раза в первый год и в 2,4 раза во второй** (17–41 тыс. шт.), а электро — всего на 8–13 %. Решение о мощностях ДВС — это и есть решение о сценарии.
2. **Гипотеза о покупках на опережение по агрегатным данным не подтверждается.** Отношение продаж IV кв. 2025 г. к I кв. 2026 г. по рынку (1,36) совпадает с сезонным профилем (1,37), по ДВС даже ниже (1,20). Сценарий «опережение» почти не отличается от «спад сохраняется»; исключение — электро 3–3,5 т (11,9 тыс. против 9,9 тыс. шт.).
3. **Обратный стресс-тест.** Чтобы объяснить нынешний уровень ДВС (≈5,9 тыс. шт./кв. без сезонности) ставкой и инвестициями, ставка должна была бы быть ≈23 % — выше исторического максимума (21 %). С II кв. 2025 г. продажи ДВС на 12–33 % ниже, чем «положено» по модели при такой ставке, а в 2024 г. были на 24–57 % выше. Значит, ставка — не единственный и, вероятно, уже не главный ограничитель: после слома действуют другие факторы (лизинг, замещение, структура спроса). Модель ARX уже сейчас «ожидает» восстановления: ставка с лагом во II–III кв. 2026 г. — 16,6 и 15,7 %, как во второй половине 2024 г., когда ДВС продавался по 11–14 тыс. шт. в квартал.
4. **Решающее правило — по сумме продаж ДВС за II и III кв. 2026 г.** Сезонность ДВС не идентифицируется, поэтому решение принимается по сумме двух кварталов в сравнении с тем же периодом 2025 г. (11 000 шт.: 5 240 и 5 760). По модели «спад сохраняется» даёт за II–III кв. ≈12,4 тыс. шт. (+13 % г/г), «восстановление» — ≈17,0 тыс. (+55 %). Не более 13,5 тыс. (+23 %) — ставочный канал не работает, план по сценариям «спад» / «ускоренное замещение» (ДВС 17–24 тыс. шт. в год); не менее 16,0 тыс. (+45 %) — переход к «восстановлению» (35–41 тыс.); между ними — ждать IV кв. Факт одного II кв. — только предварительный сигнал (граница ≈7,7 тыс.: 6,7 тыс. при спаде и 8,7 тыс. при восстановлении), так как для III кв. оценки модели — 5,8 и 8,3 тыс.

### 6.4 Рекомендации для планирования
- **Один план с коридором мощности, а сценарии — по реакции спроса, а не по макропрогнозам.** Макросценарии ЦБ и Минэкономразвития различаются на 2–4 %, держать по ним разные планы бессмысленно. Решения по ДВС привязываются к сценариям реакции спроса (раздел 5.2) и решающему правилу по факту II–III кв. 2026 г.
- **ДВС: мощности не наращивать под восстановление заранее.** Решение привязать к сигналам:
  - продажи ДВС за II–III кв. 2026 г. в сумме не менее 16,0 тыс. шт. (+45 % г/г) — восстановление подтверждается, переход к сценарию «восстановление ДВС» (35–41 тыс. шт. в год на рынке);
  - не более 13,5 тыс. шт. (+23 % г/г) — сценарии «спад сохраняется» / «ускоренное замещение»: рынок ДВС 17–24 тыс. шт. в год, мощности ДВС переводить на электропогрузчики 3–5 т;
  - между 13,5 и 16,0 тыс. — план не менять до данных IV кв.;
  - опережающий сигнал — лизинг погрузчиков (Эксперт РА: +46,7 % в I пол. 2026 г. в деньгах).
- **Электро 1–2,5 т:** во всех сценариях +2–6 % (1–1,8 т) и +11–15 % (2–2,5 т) к последним 4 кварталам, но факт за последние 4 кв. −26…−28 % г/г, а модель на истории завышала эти сегменты на 23–30 % (этап 04, раздел 5.4). Основной объём электро: выпуск держать на уровне текущего спроса, наращивать только после подтверждения фактом. Чувствительность — к инвестициям: каждый 1 п. п. инвестиций ≈ 1,7 % продаж.
- **Электро 3–5 т:** сценарный прогноз 3–3,5 т −15–20 % — следствие старта от низкого I кв. 2026; по факту г/г сегмент стабилен (+6 % за 4 кв., −4 % в I кв.), а 3,8–5 т растёт (+52 %). Это фронт замещения ДВС — приоритет развития модельного ряда (этап 04, раздел 6.4).
- **Что отслеживать ежеквартально:**
  - решения ЦБ по ставке (влияют на ДВС через 2 квартала);
  - данные Росстата по инвестициям (электро, в том же квартале);
  - мониторинг ЦБ — инвестактивность в опте, транспорте и хранении (опережающий сигнал).

  Обновление: добавить квартал в Excel и строки в `data/external/manual/scenarios.csv` → `python update.py`.
""")

md("""
## Источники
1. Данные: `data/processed/market_long.csv`, `macro_quarterly_wide.csv`, сценарии `data/external/manual/scenarios.csv` (см. ноутбуки 01, 03, 04).
2. Банк России: решения по ключевой ставке, мониторинг предприятий (инвестиционная активность).
3. Минэкономразвития России: сценарные условия социально-экономического прогноза.
4. Росстат: инвестиции в основной капитал.
5. Эксперт РА: обзор рынка лизинга за I полугодие 2026 г. (опережающий сигнал по лизингу погрузчиков).
6. Arrow K.J., Harris T., Marschak J. Optimal inventory policy // Econometrica. 1951. Vol. 19, № 3. P. 250–272 — критический квантиль для уровня плана.
""")

from _nbtools import add_asof_banner
add_asof_banner(C)
nb = nbf.v4.new_notebook(); nb.cells = C
nb.metadata.kernelspec = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, Path(__file__).with_name('05_scenarios.ipynb'))
