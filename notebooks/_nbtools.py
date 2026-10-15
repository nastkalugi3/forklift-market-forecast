"""Общие вставки для ноутбуков: предупреждение об устаревшем тексте выводов."""
import nbformat as nbf

CONCLUSIONS_ASOF = "2026Q1"   # по каким данным написан текст выводов во всех ноутбуках

BANNER = '''
from IPython.display import Markdown, display
from silant_data import load_market as _lm
_last = str(_lm(ROOT / 'data/raw/silant_forklift_market.xlsx')[0].quarter.max())
if _last != "{asof}":
    display(Markdown(f"> ⚠️ **Текст выводов ниже написан по данным до {asof}**, а в файле данные уже до **{{_last}}**. "
                     "Все таблицы и графики выше пересчитаны автоматически; конкретные цифры в тексте могли устареть — "
                     "актуальные значения см. в таблицах выше, в `reports/summary.html` и `reports/silant_forecast.xlsx`."))
else:
    print("Текст выводов соответствует текущим данным (до {asof}).")
'''


def add_asof_banner(cells):
    """Вставляет код-ячейку перед первым markdown-разделом «Выводы»."""
    for i, c in enumerate(cells):
        if c.cell_type == "markdown" and c.source.lstrip().startswith("## ") and "Выводы" in c.source.split("\n")[0]:
            cells.insert(i, nbf.v4.new_code_cell(BANNER.format(asof=CONCLUSIONS_ASOF).strip()))
            break
    return standardize(cells)


# ---------------------------------------------------------------- единый стандарт оформления ноутбуков
# (Rule et al., «Ten simple rules for writing and sharing computational analyses in Jupyter Notebooks», 2019):
# шапка с целью, входами, выходами и содержанием; пояснение перед каждым блоком кода; выводы в конце.
PIPE = {
    '01': dict(prev='—', next='02 — тренды и сезонность',
               inp='`data/raw/silant_forklift_market.xlsx` — файл заказчика',
               out='`data/processed/market_long.csv`, `quality_*.csv`; графики `reports/figures/01–08_*.png`'),
    '02': dict(prev='01 — данные', next='03 — внешние факторы',
               inp='`data/processed/market_long.csv`',
               out='`data/processed/trend_summary.csv`, `seasonal_factors.csv`; графики `reports/figures/10–16_*.png`'),
    '03': dict(prev='02 — тренды и сезонность', next='04 — прогноз на 4 квартала',
               inp='`data/external/raw/*` (ряды Банка России), `data/external/manual/*` (Росстат, прогнозы), `data/processed/market_long.csv`',
               out='`data/processed/macro_quarterly_long.csv`, `macro_quarterly_wide.csv`, `macro_forecasts.csv`, `factor_lag_check.csv`, `factor_cny_vs_usd_check.csv`, `factors_selected.csv`; графики `20–23_*.png`'),
    '04': dict(prev='03 — внешние факторы', next='05 — сценарии',
               inp='`data/processed/market_long.csv`, `macro_quarterly_wide.csv`; `data/external/manual/scenarios.csv` (ставка, если факта нет)',
               out='`data/processed/backtest_rolling_origin.csv`, `interval_calibration.json`, `forecast_4q*.csv`, `forecast_bias_backtest.csv`, `segment_yoy_check.csv`; графики `30–33_*.png`'),
    '05': dict(prev='04 — прогноз на 4 квартала', next='06 — прогноз на 8 кварталов',
               inp='`data/processed/market_long.csv`, `macro_quarterly_wide.csv`, `interval_calibration.json`; `data/external/manual/scenarios.csv`',
               out='`data/processed/scenario_*.csv`; графики `40–44_*.png`'),
    '06': dict(prev='05 — сценарии', next='— (итоги собирает `update.py`: `reports/summary.html`, Excel)',
               inp='`data/processed/market_long.csv`, `macro_quarterly_wide.csv`, `interval_calibration.json`; `data/external/manual/scenarios.csv`',
               out='`data/processed/long_forecast_*.csv`; графики `50–53_*.png`'),
}
AFTER = {
    '01': {'import sys\nimport warnings':
           '**Вывод по ячейке настройки.** Ячейка ничего не печатает: она подключает библиотеки, задаёт пути и единый стиль графиков, фиксирует seed (42) и создаёт папки `data/processed/` и `reports/figures/`. Дальше ноутбук читает только файл заказчика `data/raw/silant_forklift_market.xlsx`.'},
    '05': {'chg = pd.DataFrame(index=order)':
           '**Вывод по таблице выше.** Сценарии по ставке различаются мало. Консервативный сценарий даёт по рынку −2 % к базовому в первом году и −3 % во втором, оптимистичный +1 % и +2 %. ДВС реагирует сильнее (−3…−5 % и 0…+2 %), электро почти не реагирует (около ±1 %). Рост базового сценария ко второму году (рынок +25 %, ДВС +51 %, электро +7 % к последним 4 кварталам) почти не зависит от выбора сценария: он задан самой моделью, а не ставкой.'},
    '06': {'out = {}\nfor per in [P1, P2]:':
           '**Вывод по таблице выше.** В первом году рынок 85,5 / 87,2 / 87,8 тыс. шт. (консервативный / базовый / оптимистичный), во втором 92,8 / 95,2 / 96,8. Разница между сценариями — единицы процентов, а 80 %-ный интервал слоя L3 базового сценария во втором году (53,5–216,1 тыс.) примерно в 2,7 раза шире, чем в первом (61,1–120,6). Второй год можно использовать как диапазон, а не как точечный прогноз.'},
}
SETUP_MD = ('## 0. Настройка\n'
            'Импорт библиотек, пути (относительно корня проекта), параметры и фиксированный seed для случайных чисел. '
            'Общие функции вынесены в модули `src/`, чтобы ноутбук содержал только логику анализа.')
BANNER_MD = ('### Проверка актуальности выводов\n'
             'Текст выводов ниже написан по данным до I кв. 2026 г. Если в файле появился новый квартал, эта ячейка выведет предупреждение; '
             'таблицы и графики выше при этом пересчитываются автоматически.')
EXPLAIN = {
    '02': {'fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)':
           'Те же оценки на графике: тренд за квартал и величина ступеньки во II кв. 2025 с 90 %-ными доверительными интервалами по каждому ряду.'},
    '03': {'H = pd.DataFrame(heat).T':
           'Тепловая карта корреляций «фактор × лаг» для рынка, ДВС и электро: знак и сила связи при разных задержках. Для сравнения приведён порог, который даёт случайный инерционный ряд.',
           'fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))':
           'Прогнозы ЦБ, Минэка и аналитиков по выбранным факторам на графике.'},
    '04': {"eh = rec[rec.node.isin(KEY)]":
           'Сводка ошибок: средняя ошибка каждой модели по рынку, типам, сегментам и доле электро, выигрыш относительно наивного прогноза и ошибка по горизонту.',
           'qt = summ[summ.node.isin(KEY)].copy()':
           'Итоговая таблица прогноза на 4 квартала по рынку и типам: медиана, 80 %- и 95 %-ные интервалы, изменение к последним 4 кварталам факта.',
           'fig, axes = plt.subplots(1, 3, figsize=(15, 4))':
           'Веерный график: факт и прогноз с интервалами для рынка, ДВС и электро.',
           'segs = [b for b in order if b not in KEY]':
           'Прогноз по сегментам за 4 квартала: медиана и интервалы в сравнении с фактом последних 4 кварталов.',
           'fig, ax = plt.subplots(figsize=(9, 3.8))':
           'Доля электропогрузчиков на графике: факт и прогноз с интервалами.'},
    '05': {'# проверка двухфакторной ARX':
           'Проверка двухфакторной модели (ставка + инвестиции) на истории. Это условный прогноз: будущие значения факторов считаются известными, поэтому оценка точности оптимистична.',
           'chg = pd.DataFrame(index=order)':
           'Изменение прогноза по сценариям относительно факта последних 4 кварталов.',
           "ci = R[R['ряд'].isin(KEY)].copy()":
           'Следующая таблица: медиана и 80 %-ный интервал для рынка, ДВС и электро по каждому сценарию и каждому году. Её нужно читать вместе с таблицей выше: сценарии отличаются на единицы процентов, а интервалы — на десятки.',
           'fig, axes = plt.subplots(1, 3, figsize=(15, 4))':
           'Сценарии на графике: медианы трёх сценариев и 80 %-ный интервал базового сценария.',
           'segs = [n for n in order if n not in KEY]':
           'Сценарии по сегментам.',
           'fig, ax = plt.subplots(figsize=(9, 3.6))':
           'Доля электропогрузчиков по сценариям на графике.',
           'spread = M.pivot_table(':
           'Насколько различаются сценарии: разница медиан при комбинации моделей и при использовании только двухфакторной ARX.'},
    '06': {'fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))':
           'Веерные графики на 8 кварталов: вложенные интервалы слоёв L1–L4 и медианы трёх сценариев.',
           "d4 = T[(T['слой'] == 'L4')":
           'Следующая таблица: 80 %-ный интервал второго года по слою L4, где каждая траектория целиком берётся из одной модели (наивной, ETS или ARX), то есть добавлена неопределённость «восстановятся ли ДВС».',
           'fig, ax = plt.subplots(figsize=(9, 3.6))':
           'Доля электропогрузчиков на 8 кварталов на графике.'},
}


def _norm(t):
    # the generated code uses double quotes, the keys below may use single ones: compare without the difference
    return t.replace("'", '"')


def standardize(cells):
    """Приводит ноутбук к единому стандарту: шапка (место в пайплайне, вход, выход, содержание),
    раздел «0. Настройка», пояснение перед каждым блоком кода. Идемпотентна."""
    import re
    head = cells[0].source
    m = re.search(r'СИЛАНТ · (\d\d)\.', head)
    if not m:
        return cells
    num = m.group(1); p = PIPE[num]
    # пояснения перед кодом
    out = []
    prev_code = False        # the previous source cell was code (a conclusion cell added after it does not count)
    for i, c in enumerate(cells):
        if c.cell_type == 'code' and i == 1:
            out.append(nbf.v4.new_markdown_cell(SETUP_MD))
        elif c.cell_type == 'code' and prev_code:
            src = c.source.strip()
            if src.startswith('from IPython.display import Markdown'):
                out.append(nbf.v4.new_markdown_cell(BANNER_MD))
            else:
                txt = next((t for k, t in EXPLAIN.get(num, {}).items() if _norm(src).startswith(_norm(k))), None)
                if txt:
                    out.append(nbf.v4.new_markdown_cell(txt))
        out.append(c)
        prev_code = c.cell_type == 'code'
        if c.cell_type == 'code':
            after = next((t for k, t in AFTER.get(num, {}).items() if _norm(c.source.strip()).startswith(_norm(k))), None)
            if after:
                out.append(nbf.v4.new_markdown_cell(after))
    # шапка
    add = [f'**Место в пайплайне:** шаг {int(num)} из 6. Предыдущий: {p["prev"]}. Следующий: {p["next"]}. '
           'Запуск всех шагов: `python update.py`.']
    if '**Вход:**' not in head:
        add.append(f'**Вход:** {p["inp"]}.')
    if '**Выход:**' not in head:
        add.append(f'**Выход:** {p["out"]}.')
    secs = [c.source.split('\n')[0][3:].strip() for c in out if c.cell_type == 'markdown' and c.source.startswith('## ')]
    add.append('**Содержание:** ' + ' · '.join(secs))
    if '**Место в пайплайне:**' not in head:
        out[0] = nbf.v4.new_markdown_cell(head.rstrip() + '\n\n---\n' + '\n\n'.join(add))
    cells[:] = out
    return cells
