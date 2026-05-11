# case_6

Метрические методы регрессии: Надарая–Ватсон (fixed/variable) и LOWESS.

- Условие: `../docs/case_6/кейс 6.pdf`
- Отчёт: [`REPORT.md`](REPORT.md)
- Код: `../src/case_6`
- Тесты: `../tests/case_6` (20 unit-тестов)
- Демо-ноутбук: `../notebooks/case_6/case_6_demo.ipynb`
- Артефакты отчёта: `../report_outputs/{figures,tables,latex_fragments}`

## Запуск

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python3 -m pytest tests/case_6 -q                    # тесты
PYTHONPATH=src python3 generate_report_assets.py     # перегенерация figures/tables
python3 -m jupyter notebook notebooks/case_6/case_6_demo.ipynb
```
