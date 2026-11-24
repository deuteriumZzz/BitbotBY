# Жизненный цикл модели SAC

```
данные → features_v2 → train_sac_v2 (walk-forward) → candidate → validated → paper → shadow → canary → production
                                                          └─► rejected / retired (откат: rollback)
```

## Стадии и что нужно для перехода (`backend/bitbot/ml/model_registry.py`)
| Переход | Требования |
|---|---|
| candidate → validated | walk-forward пройден (≥ 3 фолдов), указан `dataset_version` |
| validated → paper | `approved_by` (человек) |
| paper → shadow | `approved_by` + ≥ 14 дней на paper |
| shadow → canary | `approved_by` + ≥ 14 дней на shadow |
| canary → production | `approved_by` + ≥ 14 дней на canary; прежняя production → retired (резерв для отката) |

Пропускать стадии нельзя. Понижение и отклонение — всегда, с причиной. Пороги и сроки — предложения владельцу.

## Допуск к торговле
`SAC_REQUIRE_APPROVAL=true` (по умолчанию): Paper исполняет стадии paper/canary/production, Live — только production; файл модели без записи в реестре не торгует. Canary в Live не исполняется, пока нет лимита доли капитала в исполнителе.

## Обучение
- Среда `env_v2`: действие — целевая экспозиция [−1, 1], шорт, комиссия, проскальзывание, funding, исполнение по open следующей свечи; `terminated` (банкротство) отделён от `truncated` (конец данных).
- Признаки `features_v2`: причинные, проверены тестом на look-ahead; нормализация только по train.
- Запуск: `python -m bitbot.ml.train_sac_v2 --csv … --symbol … --timeframe 15m --timesteps N --folds 4` или контейнер `docker compose --profile train run --rm trainer` (свои лимиты CPU/RAM, приоритет ниже).
- Вердикт ворот (`promotion_verdict`) — информация для реестра, не разрешение. На случайном блуждании ворота не проходятся (тест).

## Инференс
`SACV2Signal` подаёт модели то же наблюдение, на котором она училась, включая текущую позицию. Подключается флагом `SAC_V2_ENABLED`. Экспорт актора в ONNX с проверкой совпадения: `python -m bitbot.ml.onnx_export model.zip`.

## Что делает человек
Выбирает параметры среды (шорт, издержки), запускает боевое обучение, решает каждое продвижение, включает флаги. Автоматически модель не продвигается и production не заменяется.
