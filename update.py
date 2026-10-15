#!/usr/bin/env python
"""СИЛАНТ — обновление прогноза одной командой.

  python update.py                      пересчитать всё на текущем файле data/raw/…xlsx
  python update.py --xlsx НОВЫЙ.xlsx    подставить новый файл заказчика (старый уходит в архив) и пересчитать
  python update.py --fetch              сначала обновить ряды Банка России (нужен доступ к cbr.ru)
  python update.py --check              только проверить файл, ничего не пересчитывать
  python update.py --force              продолжить, даже если проверка файла нашла ошибки

Результат:
  reports/summary.html          сводка: прогноз, сценарии, доля электро, проверки, изменения к прошлому запуску
  reports/silant_forecast.xlsx   все ключевые таблицы
  reports/0X_*.html, reports/figures/*.png   подробные отчёты и графики
  data/processed/*.csv          таблицы результатов;  data/runs/<квартал>__<время>/ — архив запуска
  reports/update_log.txt        журнал
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from silant_pipeline import (NOTEBOOKS, RAW_NAME, archive_run, backtest_new_quarter, build_excel,  # noqa: E402
                             build_summary_html, check_external, diff_with_previous, latest_run, run_notebooks,
                             validate_market)

LOG = ROOT / "reports/update_log.txt"


def log(msg=""):
    print(msg, flush=True)
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    ap = argparse.ArgumentParser(description="Обновление прогноза СИЛАНТ", formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--xlsx", type=Path, help="новый файл заказчика (тот же формат, новый квартал — столбец справа)")
    ap.add_argument("--fetch", action="store_true", help="обновить ряды ЦБ перед пересчётом")
    ap.add_argument("--check", action="store_true", help="только проверка файла")
    ap.add_argument("--force", action="store_true", help="не останавливаться на ошибках проверки")
    a = ap.parse_args()

    t0 = time.time()
    raw = ROOT / "data/raw" / RAW_NAME
    log(f"\n===== Обновление {dt.datetime.now():%Y-%m-%d %H:%M:%S} =====")

    # 1. предыдущее состояние
    prev = latest_run(ROOT)
    prev_long = pd.read_csv(prev / "market_long.csv") if prev and (prev / "market_long.csv").exists() else (
        pd.read_csv(ROOT / "data/processed/market_long.csv") if (ROOT / "data/processed/market_long.csv").exists() else None)
    log(f"Прошлый запуск: {prev.name if prev else 'нет (первый запуск через update.py)'}")

    # 2. проверка файла
    src = a.xlsx.resolve() if a.xlsx else raw
    if not src.exists():
        log(f"✗ Файл не найден: {src}")
        sys.exit(2)
    log(f"[1/6] Проверка файла: {src.name}")
    chk = validate_market(src, prev_long)
    for i in chk["info"]:
        log(f"  · {i}")
    for w in chk["warnings"]:
        log(f"  ⚠️ {w}")
    for e in chk["errors"]:
        log(f"  ✗ {e}")
    if chk["errors"] and not a.force:
        log("Остановлено: исправьте файл или запустите с --force.")
        sys.exit(1)
    if a.check:
        log("Проверка завершена (--check).")
        return

    # 3. установка нового файла (перед этим — резервная копия, чтобы откатиться при сбое пересчёта)
    import tempfile
    backup = Path(tempfile.mkdtemp(prefix="silant_backup_"))
    for d_ in ("data/processed", "reports"):
        if (ROOT / d_).exists():
            shutil.copytree(ROOT / d_, backup / d_)
    if raw.exists():
        shutil.copy2(raw, backup / raw.name)
    if a.xlsx and src != raw.resolve():
        arch = ROOT / "data/raw/archive"
        arch.mkdir(parents=True, exist_ok=True)
        if raw.exists():
            shutil.copy2(raw, arch / f"{dt.datetime.now():%Y%m%d-%H%M%S}__{raw.name}")
        shutil.copy2(src, raw)
        log("[2/6] Новый файл установлен в data/raw/ (прежний сохранён в data/raw/archive/)")
    else:
        log("[2/6] Используется текущий файл data/raw/")

    # 4. внешние данные
    if a.fetch:
        log("[3/6] Обновление рядов Банка России…")
        r = subprocess.run([sys.executable, str(ROOT / "src/fetch_external.py")], capture_output=True, text=True, check=False)
        for line in (r.stdout + r.stderr).strip().splitlines():
            log(f"  {line}")
    else:
        log("[3/6] Внешние данные: используется сохранённый снимок (обновить: --fetch)")
    ext = check_external(ROOT, chk["last"])
    for m in ext:
        log(f"  ⚠️ {m}")
    chk["external"] = ext

    # 5. ноутбуки
    log(f"[4/6] Пересчёт {len(NOTEBOOKS)} ноутбуков…")
    try:
        nb_res = run_notebooks(ROOT, log=log)
    except Exception as e:  # noqa: BLE001
        log(f"✗ {e}")
        # откат: исходный файл, таблицы и отчёты возвращаются в состояние до запуска
        for d_ in ("data/processed", "reports"):
            if (backup / d_).exists():
                shutil.rmtree(ROOT / d_, ignore_errors=True)
                shutil.copytree(backup / d_, ROOT / d_)
        if (backup / raw.name).exists():
            shutil.copy2(backup / raw.name, raw)
        log("Остановлено. Файл данных, таблицы и отчёты возвращены к состоянию до запуска; архив data/runs/ не тронут.")
        sys.exit(3)
    shutil.rmtree(backup, ignore_errors=True)

    # 6. отчёты
    log("[5/6] Сводка и Excel")
    last = chk["last"]
    meta = {"last_quarter": str(last), "horizon_4q": f"{last + 1}–{last + 4}", "horizon_8q": f"{last + 5}–{last + 8}",
            "run_time": f"{dt.datetime.now():%Y-%m-%d %H:%M}", "source_file": src.name,
            "macro_snapshot": (ROOT / "data/external/raw/_snapshot.txt").read_text(encoding="utf-8").strip()
            if (ROOT / "data/external/raw/_snapshot.txt").exists() else "?",
            "duration_min": (time.time() - t0) / 60, "new_quarters": [str(q) for q in chk.get("new_quarters", [])]}
    diff = diff_with_previous(ROOT, prev)
    oos = backtest_new_quarter(ROOT, prev, chk.get("new_quarters", []))
    build_excel(ROOT, ROOT / "reports/silant_forecast.xlsx", meta, chk, diff, oos)
    build_summary_html(ROOT, ROOT / "reports/summary.html", meta, chk, diff, oos, nb_res)
    d = archive_run(ROOT, meta, raw)
    log(f"[6/6] Архив запуска: data/runs/{d.name}")
    log(f"Готово за {(time.time() - t0) / 60:.1f} мин. Откройте reports/summary.html")


if __name__ == "__main__":
    main()
