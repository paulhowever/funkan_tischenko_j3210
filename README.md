# funkan

Репозиторий подготовлен под полную Python-реализацию кейсов `case_4` и `case_6`.

## Структура

- `src/` — исходный код
- `tests/` — тесты
- `notebooks/` — Jupyter-ноутбуки для исследования и демонстрации
- `docs/` — условия кейсов (PDF)

## Кейсы

- `src/case_4`, `tests/case_4`, `notebooks/case_4`, `docs/case_4`
- `src/case_6`, `tests/case_6`, `notebooks/case_6`, `docs/case_6`

## Быстрый старт

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install pytest ruff black mypy jupyter
```

## Проверки качества

```bash
ruff check .
black --check .
mypy src
pytest
```

## Отчеты

- `notebooks/case_4/report_case_4.ipynb` — полный экспериментальный отчет по функциональной регрессии.
- `notebooks/case_6/report_case_6.ipynb` — полный экспериментальный отчет по метрическим методам.
