"""Шаги торгового цикла по порядку: скан → защиты → сопровождение → сигналы → ордер → конец.

Шаг — `async run(bot, ctx) -> CycleContext | None`. None — шаг сам закончил
цикл (сработала защита, слоты заняты) и уже выждал паузу. Имена шагов совпадают
с узлами экрана «Конвейер» в приложении.
"""

from bitbot.workers.steps import finish, guards, order, scan, signals, track

CYCLE = (scan.run, guards.run, track.run, signals.run, order.run, finish.run)
