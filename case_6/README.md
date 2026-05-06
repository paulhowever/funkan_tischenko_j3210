# case_6

Реализация метрических методов регрессии (Nadaraya-Watson и LOWESS).

- Условие: `../docs/case_6/кейс 6.pdf`
- Основной код: `../src/case_6`
- Тесты: `../tests/case_6`
- Ноутбуки: `../notebooks/case_6`

## Что уже есть

- ядра: `gaussian`, `triangular`, `epanechnikov`, `quartic`
- Nadaraya-Watson с фиксированным окном
- Nadaraya-Watson с переменным окном (`k` соседей)
- робастный LOWESS с итеративным перевзвешиванием
- LOO-подбор параметров `h`, `k` и ядра
- синтетический сценарий сравнения моделей и сценарий порога выигрыша LOWESS
- сравнение влияния выбора ядра и ширины окна
- бенчмарк на реальных датасетах: `Diabetes`, `California Housing`
- метрики `MAE`, `RMSE`, `R2`

## Где смотреть результаты

- Ноутбук отчета: `../notebooks/case_6/report_case_6.ipynb`

## Что проверяется по заданию

- fixed-window и variable-window Nadaraya-Watson;
- робастный LOWESS и веса `gamma`;
- сравнение не менее трех ядер;
- зависимость ошибки от параметров окна;
- устойчивость к выбросам;
- качество на реальных датасетах.

## Быстрый запуск

```bash
python3 -m pytest tests/case_6 -q
```

```bash
python3 -m jupyter lab
```
