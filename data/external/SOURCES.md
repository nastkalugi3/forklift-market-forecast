# Внешние данные: источники и порядок обновления

Снимок от **25.09.2026**. Каждое значение в `data/processed/macro_quarterly_long.csv` содержит источник, ссылку и дату.

## Банк России (первоисточник, обновляется скриптом `python src/fetch_external.py`)
| Файл | Показатель | Источник |
|---|---|---|
| raw/cbr_key_rate_changes.csv | Ключевая ставка (даты вступления в силу) | https://www.cbr.ru/hd_base/KeyRate/ ; решения: https://www.cbr.ru/press/keypr/ — **новое решение добавлять вручную** |
| raw/cbr_usdrub_quarterly.csv | Официальный курс USD/RUB | https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=01/01/2022&date_req2=…&VAL_NM_RQ=R01235 |
| manual/cbr_cnyrub_daily.csv | Официальный курс CNY/RUB по дням (с учётом номинала; выгрузка 02.10.2026 через веб-запрос, т.к. прямой доступ к cbr.ru из среды закрыт; `fetch_external.cny_rub()` перезаписывает файл при доступе) | https://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=01/01/2022&date_req2=…&VAL_NM_RQ=R01375 |
| manual/cbr_cpi_monthly.csv | Инфляция (ИПЦ, % г/г) по месяцам — для реальной ставки (выгрузка 02.10.2026; `fetch_external.cpi()`) | https://www.cbr.ru/hd_base/infl/ |
| manual/cpi_forecast.csv | Прогноз инфляции по кварталам 2026–2027 (ЦБ 24.07.2026, линейная интерполяция) | https://www.cbr.ru/Content/Document/File/194258/forecast_260724.pdf |
| raw/cbr_credit_legal_entities_monthly.csv | Кредиты юрлицам и ИП: выдачи и задолженность, РФ | https://www.cbr.ru/dataservice/ (publicationId=22, datasetId=49/50, measureId=22, element «Всего») |
| raw/cbr_monitoring_invest.csv | Мониторинг предприятий: инвестактивность и ожидания (SA) | https://www.cbr.ru/dataservice/ (publicationId=25, datasetId=76/77, measureId=119) |
| raw/cbr_bop_goods_quarterly.csv | Платёжный баланс: экспорт/импорт товаров | https://www.cbr.ru/dataservice/ (publicationId=9, datasetId=13) |

## Росстат через СМИ (ручное обновление `manual/rosstat_via_media.csv`)
rosstat.gov.ru, fedstat.ru и economy.gov.ru недоступны из среды, где собирался снимок. Квартальные оценки Росстата взяты из публикаций Интерфакса, РБК и Финмаркета (ссылка на каждую — в файле). При доступе к Росстату заменить на официальные таблицы:
- Инвестиции в основной капитал: https://rosstat.gov.ru/investment_nonfinancial
- Индекс промышленного производства: https://rosstat.gov.ru/enterprise_industrial

## Прогнозы (`manual/macro_forecasts.csv`)
- ЦБ, среднесрочный прогноз по итогам заседания 24.07.2026: https://www.cbr.ru/Content/Document/File/194258/forecast_260724.pdf
- Макроэкономический опрос ЦБ, сентябрь 2026: https://cbr.ru/statistics/ddkp/mo_br/
- Минэкономразвития, прогноз СЭР (сентябрь 2026) — по публикациям: https://www.kommersant.ru/doc/8974583 , https://rb.ru/news/minek-uhudshil-prognoz-po-investiciyam-v-rossii-v-2026-godu-pokazatel-upadyot-na-54/
- Минэкономразвития, сценарные условия (май 2026): https://www.interfax.ru/business/1089018

## Контекст без квартального ряда (`manual/context_not_quarterly.csv`)
Лизинг (Эксперт РА), складское строительство (IBC Real Estate, NF Group), импорт погрузчиков (не найден), утильсбор (не подтверждён НПА).

## Известные ограничения
1. Значения Росстата — последние найденные оценки; квартальные пересмотры могут отличаться.
2. Среднесрочный прогноз ЦБ извлечён из PDF автоматическим парсером; ставка 2026 записана как «14,5–14,6» (в извлечении стояло «14,5–14,61», вероятно из-за сноски). Перед защитой сверить с PDF.
3. Прогноз Минэка (сентябрь 2026) взят из СМИ, а не с сайта министерства.
4. Промпроизводство 3–4кв22 не найдено; 2кв22, 1–2кв24 — средние помесячных темпов (отмечены «ПРИБЛИЖЕНИЕ»).
