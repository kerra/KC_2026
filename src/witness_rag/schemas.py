from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

EDIT_TYPES = ("date_shift", "number_perturb", "entity_swap")
STANCES = ("true", "false", "neutral")
AUTHORS = ("human", "llm")
CELLS = ("H_T", "H_F", "H_N", "A_T", "A_F", "C_F", "G_F", "A_C")   # A_C = contrastive false passage ("X, not Y")
DEP_LEVELS = ("H0", "D0", "D2", "D3", "D4")
SHAPES = ("organic", "targeted")
SOURCE_PREFIXES = ("nq:", "trivia:")


# Собирает dataclass из словаря, пропуская неизвестные ключи: старые поля в замороженных файлах не ломают загрузку.
# Инпут: класс dataclass; dict строка jsonl
# Аутпут: экземпляр класса
def from_dict(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def source_kind(source: str) -> str:
    return source.split(":", 1)[0] if ":" in source else source


_GENERIC_LAST = {
    "Doctrine", "Bowl", "County", "River", "Award", "Awards", "Prize", "University", "City", "Kingdom", "Party",
    "Company", "Corporation", "Band", "Series", "Film", "Movie", "Show", "Song", "Album", "War", "Battle", "Treaty",
    "Act", "Bill", "Team", "Club", "League", "Cup", "Championship", "Games", "Island", "Islands", "Mountain",
    "Mountains", "Ocean", "Sea", "Lake", "Street", "Avenue", "Road", "Bridge", "Tower", "Building", "Palace", "Castle",
    "Church", "Cathedral", "Museum", "School", "College", "Hospital", "Airport", "Station", "Park", "Square", "Empire",
    "Republic", "Union", "State", "States", "Province", "Region", "District", "Valley", "Desert", "Forest", "Bay",
    "Canal", "Railway", "Line", "Day", "Night", "Street", "House", "Hall", "Center", "Centre", "Institute", "Group",
    "Family", "Brothers", "Sisters", "Coast", "Beach", "Point", "Falls", "Hills", "Heights", "Gardens", "Fields",
    "Ranch", "Farm", "Store", "Shop", "Market", "Hotel", "Theatre", "Theater", "Stadium", "Arena", "Temple", "Abbey",
    "Fort", "Harbor", "Harbour", "Port", "Trail", "Route", "Highway", "Tunnel", "Bank", "Society", "Association",
    "Foundation", "Council", "Committee", "Commission", "Agency", "Department", "Ministry", "Office", "Court",
    "Parliament", "Congress", "Senate", "Army", "Navy", "Force", "Library", "Gallery", "Studio", "Records", "Pictures",
    "Network", "Radio", "Television", "Press", "Times", "Post", "Journal", "Magazine", "Book", "Novel", "Story",
    "Poem", "Symphony", "Opera", "Ballet", "Musical", "Play", "Saga", "Trilogy", "Wars", "Ring", "Rings", "Stone",
    "Stones", "Crown", "Man", "Woman", "Boy", "Girl", "King", "Queen", "Prince", "Princess", "Lord", "Lady", "Doctor",
    "Captain", "General", "President", "Airlines", "Motors", "Industries", "Systems", "Technologies", "Software",
}
_ROMAN = re.compile(r"^[IVXLCDM]+$")
_STRIP = ".,;:!?\"'()"
_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
         "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_ORDINALS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
             9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
             15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth", 19: "nineteenth", 20: "twentieth"}


def _words_for(n: int) -> list[str]:
    if n < 20:
        return [_ONES[n]]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return [_TENS[tens]]
    return [f"{_TENS[tens]}-{_ONES[ones]}", f"{_TENS[tens]} {_ONES[ones]}"]


_WORD_TO_NUM = {w: n for n in range(100) for w in _words_for(n)}
_ORD_TO_NUM = {w: n for n, w in _ORDINALS.items()}


# Переводит малые числа между цифрами и словами в обе стороны ("24" >> "twenty-four", "Sixth" >> "6th"),
# чтобы проверка значения принимала любую форму записи.
# Инпут: str значение
# Аутпут: list[str] альтернативные написания (пусто, если значение не число)
def number_aliases(value: str) -> list[str]:
    v = value.strip().lower()
    if v.isdigit() and len(v) <= 2:
        return _words_for(int(v))
    key = v.replace("-", " ")
    key2 = key.replace(" ", "-") if len(key.split()) == 2 else key
    for k in (key, key2, v):
        if k in _WORD_TO_NUM:
            return [str(_WORD_TO_NUM[k])]
    if v in _ORD_TO_NUM:
        n = _ORD_TO_NUM[v]
        return [f"{n}{'th' if n % 100 in (11, 12, 13) or n % 10 not in (1, 2, 3) else ('st', 'nd', 'rd')[n % 10 - 1]}"]
    m = re.fullmatch(r"(\d{1,2})(st|nd|rd|th)", v)
    if m and int(m.group(1)) in _ORDINALS:
        return [_ORDINALS[int(m.group(1))]]
    return []


# Выводит дополнительные алиасы значения: без ведущего описателя ("planner Raymond Unwin" >> "Raymond Unwin"),
# фамилия многословного имени, числовые формы, единственное/множественное число последнего нарицательного слова.
# Инпут: str значение
# Аутпут: list[str] алиасы без самого значения
def derived_aliases(value: str) -> list[str]:
    toks = [t.strip(_STRIP) for t in value.strip().split()]
    toks = [t for t in toks if t]
    if not toks:
        return []
    out: list[str] = list(number_aliases(value))
    # singular/plural of a trailing common noun: both directions when it is lower-cased ("Horse islands" >> "Horse
    # island", "eight counties" >> "eight county"), singular only for a capitalised generic plural ("Falkland Islands"
    # >> "Falkland Island"); never for a trailing proper name ("Pyotr Ilyich Tchaikovsky" stays as it is)
    if len(toks) >= 2 and toks[-1].isalpha() and len(toks[-1]) >= 4 and not re.search(r"\d", value):
        last = toks[-1]
        head = " ".join(toks[:-1])
        plural = last.lower().endswith("s") and not last.lower().endswith("ss")
        common = last.islower() and toks[0][0].isupper()          # "Horse islands"; not "saint james" (lower-cased TriviaQA name)
        if plural and (common or last in _GENERIC_LAST):
            out.append(f"{head} {last[:-3]}y" if last.lower().endswith("ies") else f"{head} {last[:-1]}")
        elif common:
            out.append(f"{head} {last[:-1]}ies" if last.endswith("y") and last[-2] not in "aeiou" else f"{head} {last}s")
    core = toks[:]
    while len(core) > 1 and core[0][0].islower():
        core = core[1:]
    # keep the stripped core only when it is a proper name or a number ("Raymond Unwin", "1975"), never a
    # lower-cased common noun ("saint james" >> "james", "cell nucleus" >> "nucleus" would be wrong)
    if core != toks and (core[0][0].isupper() or core[0][0].isdigit()) and len(" ".join(core)) >= 3:
        out.append(" ".join(core))
    else:
        core = toks
    if 2 <= len(core) <= 4 and not re.search(r"\d", " ".join(core)) and all(t[0].isupper() for t in core):
        last = core[-1]
        if len(last) >= 4 and last.isalpha() and not _ROMAN.match(last) and last not in _GENERIC_LAST:
            out.append(last)
    return out


def _dedupe(values: list[str]) -> list[str]:
    seen, out = set(), []
    for v in values:
        key = v.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(v.strip())
    return out


# Пара вопрос + истинное и ложное значение, единица бенчмарка; непустой excluded = пара выведена из корпуса, id сохранён.
# Инпут: поля строки data/claims/pairs.jsonl
# Аутпут: объект; gold_aliases() / attack_aliases() >> list[str] алиасов для проверки ответов и текстов
@dataclass
class Pair:
    pair_id: str
    question: str
    answer_true: str
    answer_false: str
    edit_type: str
    source: str                      # "nq:<id>" | "trivia:<id>"
    topic: str = ""
    answer_true_aliases: list[str] = field(default_factory=list)
    answer_false_aliases: list[str] = field(default_factory=list)
    notes: str = ""
    excluded: str = ""               # non-empty = reason the pair is out of the corpus (ids stay stable; docs are ignored)

    def gold_aliases(self) -> list[str]:
        return _dedupe([self.answer_true, *self.answer_true_aliases, *derived_aliases(self.answer_true)])

    def attack_aliases(self) -> list[str]:
        return _dedupe([self.answer_false, *self.answer_false_aliases, *derived_aliases(self.answer_false)])

    def answer_for(self, stance: str) -> str:
        return self.answer_true if stance == "true" else self.answer_false

    def aliases_for(self, stance: str) -> list[str]:
        return self.gold_aliases() if stance == "true" else self.attack_aliases()

    # Проверяет пару: непустые поля, допустимый edit_type, разные значения, значения не утекают в вопрос, source с префиксом nq:/trivia:.
    # Инпут: нет (поля объекта)
    # Аутпут: list[str] сообщений об ошибках (пустой список = пара валидна)
    def validate(self) -> list[str]:
        errs: list[str] = []
        for name in ("pair_id", "question", "answer_true", "answer_false", "edit_type", "source"):
            if not str(getattr(self, name) or "").strip():
                errs.append(f"{name} is empty")
        if errs:
            return errs
        if self.edit_type not in EDIT_TYPES:
            errs.append(f"edit_type {self.edit_type!r} not in {EDIT_TYPES}")
        if self.answer_true.strip().lower() == self.answer_false.strip().lower():
            errs.append("answer_false equals answer_true")
        for alias in self.gold_aliases():
            if alias and alias.lower() in self.question.lower():
                errs.append(f"gold alias {alias!r} leaks into the question")
        for alias in self.attack_aliases():
            if alias and alias.lower() in self.question.lower():
                errs.append(f"attack alias {alias!r} leaks into the question")
        if not self.source.startswith(SOURCE_PREFIXES):
            errs.append(f"source must start with one of {SOURCE_PREFIXES}")
        return errs

    def to_dict(self) -> dict:
        return asdict(self)


# Строка производственного лога: запечатанная разметка документа (позиция, автор, ячейка, уровень зависимости, кластер, генератор).
# Сигналы её никогда не читают; на ней считаются исходы и оракул.
# Инпут: поля строки data/production_log.jsonl
# Аутпут: объект; validate() >> list[str] ошибок
@dataclass
class DocRecord:
    doc_id: str
    pair_id: str
    stance: str                      # true | false | neutral
    author: str                      # human | llm
    cell: str                        # see CELLS
    dependence_level: str            # H0 | D0 | D2 | D3 | D4
    cluster_id: str                  # ground-truth latent source
    model: Optional[str] = None      # generator id (llm) or paraphraser id (D3); None for human
    genre: Optional[str] = None
    shape: Optional[str] = None      # organic | targeted (LLM cells and G_F); None for human
    prompt_id: Optional[str] = None
    temperature: Optional[float] = None
    seed: Optional[int] = None
    length_target: Optional[int] = None
    window_start: Optional[int] = None   # token offset of the 100-word window inside the full generation
    salvaged: Optional[list] = None      # edits applied after the retries ("substituted_2_mentions", ...); None = clean
    parent_doc: Optional[str] = None # required for D3 / D4 / H_F (value-substituted copy of H_T)
    lang: str = "en"
    gen_date: str = ""

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.stance not in STANCES:
            errs.append(f"{self.doc_id}: stance {self.stance!r}")
        if self.author not in AUTHORS:
            errs.append(f"{self.doc_id}: author {self.author!r}")
        if self.cell not in CELLS:
            errs.append(f"{self.doc_id}: cell {self.cell!r}")
        if self.dependence_level not in DEP_LEVELS:
            errs.append(f"{self.doc_id}: dependence_level {self.dependence_level!r}")
        if self.shape is not None and self.shape not in SHAPES:
            errs.append(f"{self.doc_id}: shape {self.shape!r}")
        if self.dependence_level in ("D3", "D4") and not self.parent_doc:
            errs.append(f"{self.doc_id}: {self.dependence_level} requires parent_doc")
        if self.author == "llm" and not self.model:
            errs.append(f"{self.doc_id}: llm doc without model")
        if self.author == "human" and self.model:
            errs.append(f"{self.doc_id}: human doc with model")
        return errs

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Document:
    doc_id: str
    text: str
    n_words: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Cluster:
    cluster_id: str
    pair_id: str
    stance: str
    size: int
    members: list[str]
    root: str
    kind: str

    def to_dict(self) -> dict:
        return asdict(self)
