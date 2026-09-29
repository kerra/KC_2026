from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from ..io import read_yaml, resolve, sha256_text
from .base import Context, ContextDoc, Signal

_FUNCTION_WORDS = set("""the of and to in a is that for it as was with be by on not he i this are or his from at
which but have an had they you were their one all we can her has there been if more when will would who so no
which what about out up its into than them then some could time these two may only other new do any very like
should over such after most also made many before must through back years where much your way well down""".split())
_STOP = set("the a an of in on at to for and or is was were are be by with from as that this which who whom whose "
            "what when where why how did does do has have had it its his her their he she they you we i".split())
_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z']+")
_TOKEN = re.compile(r"[a-z0-9']+")


# ----------------------------------------------------------------------------- scorers
# 17 поверхностных стилометрических признаков: статистики длин предложений, разнообразие лексики, доли служебных слов,
# пунктуации, цифр и заглавных букв.
# Инпут: str текст (features) или list[str] тексты (score)
# Аутпут: np.ndarray признаков
class StylometricScorer:
    name = "stylometric"
    needs_question = False
    n_features = 17

    def features(self, text: str) -> np.ndarray:
        sents = [s for s in _SENT.split(text.strip()) if s.strip()] or [text]
        words = _WORD.findall(text)
        n_w = max(1, len(words))
        lens = np.array([len(_WORD.findall(s)) for s in sents], dtype=float)
        lower = [w.lower() for w in words]
        counts: dict[str, int] = {}
        for w in lower:
            counts[w] = counts.get(w, 0) + 1
        n_chars = max(1, len(text))
        feats = [
            lens.mean(), lens.std(), (lens.max() - lens.min()) if len(lens) else 0.0,
            len(counts) / n_w,
            sum(1 for w in lower if w in _FUNCTION_WORDS) / n_w,
            sum(1 for c in counts.values() if c == 1) / n_w,
            float(np.mean([len(w) for w in words])) if words else 0.0,
            sum(1 for w in words if len(w) >= 9) / n_w,
            text.count(",") / n_w, text.count(";") / n_w, text.count(":") / n_w,
            (text.count("—") + text.count(" - ") + text.count("–")) / n_w,
            text.count("!") / n_w, text.count("?") / n_w, text.count('"') / n_w,
            sum(c.isdigit() for c in text) / n_chars,
            sum(1 for w in words[1:] if w[0].isupper()) / n_w,
        ]
        return np.array(feats, dtype=float)

    def score(self, texts: list[str], questions: list[str] | None = None) -> np.ndarray:
        return np.stack([self.features(t) for t in texts])


# 8 признаков ответности на запрос: пересечение с вопросом, самая длинная общая n-грамма, шаблонные фразы ответа,
# тег источника в начале ("CNN news:"), вопросительные знаки, плотность предложений. Видит текст и вопрос, никогда ответ.
# Инпут: str текст; str вопрос
# Аутпут: np.ndarray из 8 признаков
class ShapeScorer:
    name = "shape"
    needs_question = True
    n_features = 8
    _TEMPLATES = [re.compile(p, re.I) for p in (
        r"\bthe answer\b", r"\bis indeed\b", r"\bit is clear\b", r"\btherefore\b", r"\brather than\b",
        r"\bin fact\b", r"\bsafe to say\b", r"\bcorrect answer\b", r"\bnot \w+(?: \w+){0,3},? but\b",
        r"\baccording to\b", r"\bthe question\b")]
    _SOURCE_TAG = re.compile(
        r"^\s*(CNN news|According to|Reports? |A recent|An? \w+ (report|archive|database|guide|retrospective|"
        r"catalogue|summary|profile|timeline|feature|article))", re.I)

    @staticmethod
    def _content(text: str) -> list[str]:
        return [w for w in _TOKEN.findall(text.lower()) if w not in _STOP]

    def features(self, text: str, question: str) -> np.ndarray:
        q = self._content(question)
        t_tokens = _TOKEN.findall(text.lower())
        t = set(t_tokens)
        q_set = set(q)
        inter = len(q_set & t)
        recall = inter / len(q_set) if q_set else 0.0
        jacc = inter / len(q_set | t) if (q_set | t) else 0.0
        sents = [s for s in _SENT.split(text.strip()) if s.strip()] or [text]
        first = set(_TOKEN.findall(sents[0].lower()))
        first_recall = len(q_set & first) / len(q_set) if q_set else 0.0
        longest = 0
        q_full = _TOKEN.findall(question.lower())
        joined = " " + " ".join(t_tokens) + " "
        for n in range(min(len(q_full), 12), 1, -1):
            if any((" " + " ".join(q_full[i:i + n]) + " ") in joined for i in range(len(q_full) - n + 1)):
                longest = n
                break
        n_words = max(1, len(t_tokens))
        feats = [
            recall, jacc, first_recall,
            longest / len(q_full) if q_full else 0.0,
            1.0 if self._SOURCE_TAG.search(text) else 0.0,
            100.0 * sum(len(p.findall(text)) for p in self._TEMPLATES) / n_words,
            1.0 if "?" in text else 0.0,
            100.0 * len(sents) / n_words,
        ]
        return np.array(feats, dtype=float)

    def score(self, texts: list[str], questions: list[str] | None = None) -> np.ndarray:
        if questions is None:
            raise ValueError("ShapeScorer needs the question for every text")
        return np.stack([self.features(t, q) for t, q in zip(texts, questions)])


# Плотность маркеров AI-текста (Wikipedia: Signs of AI writing) на 100 слов по группам из configs/ai_markers.yaml плюс общая плотность.
# Инпут: str путь к YAML маркеров или None
# Аутпут: features(text) >> np.ndarray плотностей по группам и суммарной
class LexiconScorer:
    name = "lexicon"
    needs_question = False

    def __init__(self, path: str | None = None):
        cfg = read_yaml(resolve(path or "configs/ai_markers.yaml"))
        self.patterns: dict[str, re.Pattern] = {}
        for group, phrases in cfg["phrases"].items():
            alts = sorted({p.strip() for p in phrases if p and p.strip()}, key=len, reverse=True)
            self.patterns[group] = re.compile(r"(?<![A-Za-z])(?:" + "|".join(re.escape(a) for a in alts) + r")(?![A-Za-z])", re.I)
        for group, rx in cfg["regexes"].items():
            self.patterns[group] = re.compile(rx, re.I)
        self.groups = list(self.patterns.keys())
        self.n_features = len(self.groups) + 1        # + total density

    def counts(self, text: str) -> dict[str, int]:
        return {g: len(p.findall(text)) for g, p in self.patterns.items()}

    def features(self, text: str) -> np.ndarray:
        n_w = max(1, len(_WORD.findall(text)))
        c = self.counts(text)
        rates = [100.0 * c[g] / n_w for g in self.groups]
        return np.array(rates + [sum(rates)], dtype=float)

    def score(self, texts: list[str], questions: list[str] | None = None) -> np.ndarray:
        return np.stack([self.features(t) for t in texts])


# Zero-shot оценка Binoculars (Hans et al. 2024): лог-перплексия текста под моделью-наблюдателем, делённая на кросс-энтропию
# наблюдатель/исполнитель; ниже = более машинно. Две малые модели с общим токенизатором.
# Инпут: str пути наблюдателя и исполнителя; int максимум токенов; str устройство
# Аутпут: score(texts) >> np.ndarray формы (n, 1)
class BinocularsScorer:
    name = "binoculars"
    needs_question = False
    n_features = 1

    def __init__(self, observer: str, performer: str, max_tokens: int = 512, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(observer)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.observer = AutoModelForCausalLM.from_pretrained(observer, torch_dtype=dtype).to(self.device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(performer, torch_dtype=dtype).to(self.device).eval()
        self.max_tokens = max_tokens

    def score_one(self, text: str) -> float:
        torch = self.torch
        enc = self.tok(text, return_tensors="pt", truncation=True, max_length=self.max_tokens).to(self.device)
        with torch.no_grad():
            lo = self.observer(**enc).logits[:, :-1].float()
            lp = self.performer(**enc).logits[:, :-1].float()
        targets = enc["input_ids"][:, 1:]
        log_q = torch.log_softmax(lo, dim=-1)
        log_ppl = -log_q.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
        x_ent = -(torch.softmax(lp, dim=-1) * log_q).sum(-1).mean()
        return float(log_ppl / x_ent)

    def score(self, texts: list[str], questions: list[str] | None = None) -> np.ndarray:
        return np.array([[self.score_one(t)] for t in texts], dtype=float)


# Создаёт скорер формы по имени метода.
# Инпут: str метод (binoculars | stylometric | shape | lexicon); dict конфиг signals.form
# Аутпут: объект скорера с полями name, needs_question и методом score()
def make_scorer(method: str, fcfg: dict):
    if method == "binoculars":
        return BinocularsScorer(fcfg["observer"], fcfg["performer"], int(fcfg.get("max_tokens", 512)))
    if method == "stylometric":
        return StylometricScorer()
    if method == "shape":
        return ShapeScorer()
    if method == "lexicon":
        return LexiconScorer()
    raise ValueError(f"unknown form method {method!r}")


# ----------------------------------------------------------------------------- calibration
# Логистическая регрессия на стандартизованных признаках, обучается демпфированным методом Ньютона без sklearn; сохраняется в JSON.
# Инпут: fit(X, y) >> матрица признаков и метки 0/1; l2-регуляризация
# Аутпут: predict_proba(X) >> np.ndarray вероятностей помеченного класса
class LogisticCalibrator:

    def __init__(self, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray):
        self.w, self.b, self.mean, self.std = w, b, mean, std

    @classmethod
    def fit(cls, X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 50) -> "LogisticCalibrator":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        mean, std = X.mean(axis=0), X.std(axis=0)
        std = np.where(std < 1e-12, 1.0, std)            # constant features stay 0 after centering
        Zb = np.hstack([(X - mean) / std, np.ones((X.shape[0], 1))])
        theta = np.zeros(Zb.shape[1])
        reg = np.full(Zb.shape[1], l2)
        reg[-1] = 0.0
        # errstate: on macOS/Accelerate numpy raises spurious FP flags inside matmul for tiny weights;
        # the result is checked for finiteness below instead.
        with np.errstate(all="ignore"):
            for _ in range(iters):
                z = np.clip(Zb @ theta, -30.0, 30.0)
                p = 1.0 / (1.0 + np.exp(-z))
                grad = Zb.T @ (p - y) + reg * theta
                H = (Zb * (p * (1 - p))[:, None]).T @ Zb + np.diag(reg) + 1e-6 * np.eye(len(theta))
                step = np.clip(np.linalg.solve(H, grad), -5.0, 5.0)
                theta -= step
                if np.abs(step).max() < 1e-8:
                    break
        if not np.all(np.isfinite(theta)):
            raise FloatingPointError("logistic calibrator diverged; increase l2")
        return cls(theta[:-1], float(theta[-1]), mean, std)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Z = (np.asarray(X, dtype=float) - self.mean) / self.std
        return 1.0 / (1.0 + np.exp(-np.clip(Z @ self.w + self.b, -30.0, 30.0)))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"w": self.w.tolist(), "b": self.b, "mean": self.mean.tolist(),
                                          "std": self.std.tolist()}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LogisticCalibrator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(d["w"]), float(d["b"]), np.array(d["mean"]), np.array(d["std"]))


# ----------------------------------------------------------------------------- signal
# Сигнал формы: признаки скорера >> калибратор, обученный на dev >> p(flagged); сырая оценка пассажа = 1 - p.
# Признаки кэшируются по sha256 текста (и вопроса, если скорер его использует).
# Инпут: скорер; LogisticCalibrator или None; Path кэша; str имя арма
# Аутпут: Signal; scores(ctx) >> list[float]; extras(ctx) >> p_<имя> по пассажам
class FormSignal(Signal):
    def __init__(self, scorer, calibrator: LogisticCalibrator | None, cache_path: Path | None = None, name: str = "form"):
        self.name = name
        self.scorer = scorer
        self.needs_question = bool(getattr(scorer, "needs_question", False))
        self.calibrator = calibrator
        self.cache_path = cache_path
        self.cache: dict[str, list[float]] = {}
        if cache_path and Path(cache_path).exists():
            with open(cache_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        self.cache[row["key"]] = row["features"]

    @classmethod
    def from_config(cls, fcfg: dict, cache_dir: Path | None = None, calibrator_path: Path | None = None,
                    method: str | None = None, name: str = "form") -> "FormSignal":
        method = method or fcfg.get("method", "binoculars")
        scorer = make_scorer(method, fcfg)
        calib = LogisticCalibrator.load(calibrator_path) if calibrator_path and Path(calibrator_path).exists() else None
        cache = (cache_dir / f"form_{method}_cache.jsonl") if cache_dir else None
        return cls(scorer, calib, cache, name=name)

    def _key(self, text: str, question: str | None) -> str:
        return sha256_text((question or "") + "\x1f" + text) if self.needs_question else sha256_text(text)

    def features(self, texts: list[str], questions: list[str] | None = None) -> np.ndarray:
        qs = questions if questions is not None else [None] * len(texts)
        if self.needs_question and any(q is None for q in qs):
            raise ValueError(f"{self.name} signal needs a question for every text")
        missing = [(t, q) for t, q in zip(texts, qs) if self._key(t, q) not in self.cache]
        if missing:
            feats = self.scorer.score([t for t, _ in missing], [q for _, q in missing] if self.needs_question else None)
            for (t, q), f in zip(missing, feats):
                key = self._key(t, q)
                self.cache[key] = [float(x) for x in f]
                if self.cache_path:
                    Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                    with open(self.cache_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps({"key": key, "features": self.cache[key]}) + "\n")
        return np.array([self.cache[self._key(t, q)] for t, q in zip(texts, qs)], dtype=float)

    def precompute(self, docs: list[ContextDoc]) -> None:
        if not self.needs_question:
            self.features([d.text for d in docs])

    def p_flag(self, texts: list[str], questions: list[str] | None = None) -> np.ndarray:
        if self.calibrator is None:
            raise RuntimeError(f"{self.name} signal needs a calibrator fitted on the dev split (experiments/fit_form_calibrator.py)")
        return self.calibrator.predict_proba(self.features(texts, questions))

    def scores(self, ctx: Context) -> list[float]:
        qs = [ctx.question] * len(ctx.docs) if self.needs_question else None
        return [float(1.0 - p) for p in self.p_flag(ctx.texts(), qs)]

    def extras(self, ctx: Context) -> dict:
        qs = [ctx.question] * len(ctx.docs) if self.needs_question else None
        return {f"p_{self.name}": [round(float(p), 4) for p in self.p_flag(ctx.texts(), qs)]}


# Обучает калибратор p(flagged) на dev-документах (единственное место, где сигнал формы видит метки) и сохраняет его.
# Инпут: FormSignal; list[str] тексты; list[int] метки; Path вывода; list[str] вопросы (только для shape)
# Аутпут: dict n, dev AUC, путь к калибратору
def fit_calibrator(signal: FormSignal, texts: list[str], labels: list[int], out_path: Path,
                   questions: list[str] | None = None) -> dict:
    X = signal.features(texts, questions)
    y = np.asarray(labels, dtype=float)
    calib = LogisticCalibrator.fit(X, y)
    calib.save(out_path)
    signal.calibrator = calib
    return {"n": len(texts), "auc": roc_auc(y, calib.predict_proba(X)), "path": str(out_path)}


# AUC ROC через ранги со средними рангами для равных оценок.
# Инпут: метки 0/1; оценки
# Аутпут: float (nan, если представлен один класс)
def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y, s = np.asarray(y, dtype=float), np.asarray(s, dtype=float)
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    if n_pos == 0 or n_neg == 0:
        return math.nan
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))
