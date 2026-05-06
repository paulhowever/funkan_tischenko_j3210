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
- синтетический сценарий сравнения моделей
- метрики `MAE`, `RMSE`, `R2`

## Быстрый запуск

```bash
python3 -m pytest tests/case_6 -q
```
