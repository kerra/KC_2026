from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ContextDoc:
    doc_id: str
    text: str
    sealed: dict = field(default_factory=dict, repr=False)


@dataclass
class Context:
    pair_id: str
    condition: str
    k: int
    question: str
    docs: list[ContextDoc]
    order_seed: int = 0

    def texts(self) -> list[str]:
        return [d.text for d in self.docs]

    def doc_ids(self) -> list[str]:
        return [d.doc_id for d in self.docs]


# Базовый интерфейс сигнала: одна сырая оценка в [0, 1] на пассаж (1 = полное доверие); запечатанную разметку читает только oracle.
# Инпут: Context (вопрос и упорядоченные пассажи)
# Аутпут: scores(ctx) >> list[float]; extras(ctx) >> dict диагностик для строки результата
class Signal:
    name = "base"
    requires_labels = False

    def scores(self, ctx: Context) -> list[float]:
        raise NotImplementedError

    def precompute(self, docs: list[ContextDoc]) -> None:
        return None

    def extras(self, ctx: Context) -> dict:
        return {}


class NoneSignal(Signal):
    name = "none"

    def scores(self, ctx: Context) -> list[float]:
        return [1.0] * len(ctx.docs)


# Произведение двух сигналов поэлементно (арм judge_x_dependence).
# Инпут: два Signal
# Аутпут: Signal с именем a_x_b
class ProductSignal(Signal):

    def __init__(self, a: Signal, b: Signal):
        self.a, self.b = a, b
        self.name = f"{a.name}_x_{b.name}"
        self.requires_labels = a.requires_labels or b.requires_labels

    def scores(self, ctx: Context) -> list[float]:
        return [x * y for x, y in zip(self.a.scores(ctx), self.b.scores(ctx))]

    def precompute(self, docs: list[ContextDoc]) -> None:
        self.a.precompute(docs)
        self.b.precompute(docs)

    def extras(self, ctx: Context) -> dict:
        return {**self.a.extras(ctx), **self.b.extras(ctx)}


# Переводит сырые оценки сигнала в множители внимания. clip: max(floor, min(1, s)), абсолютный порог; maxnorm: s / max(s)
# с полом floor, лучший пассаж всегда сохраняет полное внимание (нормировка кода CrAM без сдвига на минимум).
# Инпут: list[float] сырые оценки пассажей одного контекста; float пол множителя; str имя преобразования (clip | maxnorm)
# Аутпут: list[float] множители в [floor, 1] в порядке пассажей
def to_multipliers(scores: list[float], floor: float = 0.1, transform: str = "clip") -> list[float]:
    if transform == "clip":
        return [float(min(1.0, max(floor, s))) for s in scores]
    if transform == "maxnorm":
        hi = max(scores)
        if hi <= 1e-12:
            return [1.0] * len(scores)
        return [float(min(1.0, max(floor, s / hi))) for s in scores]
    raise ValueError(f"unknown transform {transform!r}")


# Фабрика армов по имени: none, oracle, judge, form / form_<method>, shape, dependence, agreement, voting и произведения a_x_b.
# Кэш признаков формы общий, кэш ридера (вердикты судьи, извлечённые ответы) свой у каждого ридера.
# Инпут: str имя арма; dict конфиг; callable генерации судьи; Path кэша признаков; Path калибратора; callable извлечения ответа; Path кэша ридера
# Аутпут: Signal
def make_signal(name: str, cfg: dict, judge_generate=None, cache_dir=None, calibrator_path=None, extract=None,
                reader_cache_dir=None) -> Signal:
    scfg = cfg["signals"]
    rdir = reader_cache_dir if reader_cache_dir is not None else cache_dir
    if "_x_" in name:
        a, b = name.split("_x_", 1)
        return ProductSignal(make_signal(a, cfg, judge_generate, cache_dir, calibrator_path, extract, rdir),
                             make_signal(b, cfg, judge_generate, cache_dir, calibrator_path, extract, rdir))
    if name in ("agreement", "voting"):
        from .agreement import AgreementSignal
        from .dependence_discount import DependenceSignal
        if extract is None:
            raise ValueError(f"{name} signal needs an extract(question, passage) callable")
        dep = DependenceSignal.from_config(scfg["dependence"]) if name == "agreement" else None
        return AgreementSignal(extract, cache_path=(rdir / "extract_cache.jsonl") if rdir else None,
                               dependence=dep, name=name)
    if name == "none":
        return NoneSignal()
    if name == "oracle":
        from .oracle import OracleSignal
        return OracleSignal()
    if name == "dependence":
        from .dependence_discount import DependenceSignal
        return DependenceSignal.from_config(scfg["dependence"])
    if name == "form":
        from .form_score import FormSignal
        return FormSignal.from_config(scfg["form"], cache_dir=cache_dir, calibrator_path=calibrator_path)
    if name.startswith("form_"):
        from pathlib import Path as _P
        from .form_score import FormSignal
        method = name[len("form_"):]
        calib = _P(calibrator_path).with_name(f"form_{method}.json") if calibrator_path is not None else None
        return FormSignal.from_config(scfg["form"], cache_dir=cache_dir, calibrator_path=calib, method=method, name=name)
    if name == "shape":
        from .form_score import FormSignal
        shape_calib = None
        if calibrator_path is not None:
            from pathlib import Path as _P
            shape_calib = _P(calibrator_path).with_name("form_shape.json")
        return FormSignal.from_config(scfg["form"], cache_dir=cache_dir, calibrator_path=shape_calib,
                                      method="shape", name="shape")
    if name == "judge":
        from .llm_judge import JudgeSignal
        if judge_generate is None:
            raise ValueError("judge signal needs a generate(prompt) callable")
        return JudgeSignal(judge_generate, cache_path=(rdir / "judge_cache.jsonl") if rdir else None)
    raise ValueError(f"unknown signal {name!r}")
