from __future__ import annotations

from .base import Context, Signal


# Оракульный сигнал: читает запечатанную разметку и даёт ложным пассажам 0, остальным 1 (потолок для сравнения, не метод).
# Инпут: Context, у документов которого заполнено поле sealed
# Аутпут: list[float] сырых оценок 0/1 в порядке пассажей
class OracleSignal(Signal):
    name = "oracle"
    requires_labels = True

    def scores(self, ctx: Context) -> list[float]:
        return [0.0 if d.sealed.get("stance") == "false" else 1.0 for d in ctx.docs]
