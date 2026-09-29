from __future__ import annotations

import re

from .coordinated import split_sentences

_MD_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__", re.S)
_MD_ITALIC = re.compile(r"(?<![\w*])\*(?!\s)([^*\n]+?)(?<!\s)\*(?!\w)|(?<![\w_])_(?!\s)([^_\n]+?)(?<!\s)_(?!\w)")
_MD_HEADING = re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+")
_MD_BULLET = re.compile(r"(?m)^[ \t]*(?:[-*•]|\d+[.)])[ \t]+")
_MD_RULE = re.compile(r"(?m)^[ \t]*(?:-{3,}|\*{3,}|_{3,})[ \t]*$")
_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_BLANKS = re.compile(r"[ \t]+\n")


# Убирает markdown-разметку (жирный, курсив, заголовки, списки, ссылки), оставляя слова; идемпотентно.
# Инпут: str текст
# Аутпут: str текст без разметки
def strip_markdown(text: str) -> str:
    t = _MD_LINK.sub(r"\1", text)
    t = _MD_BOLD.sub(lambda m: m.group(1) or m.group(2) or "", t)
    t = _MD_ITALIC.sub(lambda m: m.group(1) or m.group(2) or "", t)
    t = _MD_HEADING.sub("", t)
    t = _MD_RULE.sub("", t)
    t = _MD_BULLET.sub("", t)
    return _BLANKS.sub("\n", t).strip()


def has_markdown(text: str) -> bool:
    return bool(_MD_BOLD.search(text) or _MD_HEADING.search(text) or _MD_RULE.search(text))


# Denial of the asserted value, anchored on the value itself. Loose cue windows ("not" within 60 characters) were
# tried first and flagged tag questions ("isn't it?"), "Despite not being a primary character, Rover ..." and
# "viewers may not be aware of Spike's name"; the anchored forms below did not.
_NOT = r"(?:not|never|n't|nor|isn't|wasn't|aren't|weren't)"
_ERR = r"(?:mistakenly|erroneously|falsely|wrongly|incorrectly)"
_AUTH_VERBS = (r"(?:compos|writ|wrot|creat|direct|voic|play|starr|found|invent|discover|w[io]n|author|paint|record|releas|"
               r"s[aiu]ng|design|buil|produc|ma[dk]e|scor|penn|nam|portray|perform|host|le[ad]|coach|own|establish|develop|"
               r"publish|appear|exist|ha[vd]|h[oe]ld|serv|bec[ao]me|receiv|earn|conduct|choreograph|form|start|launch|"
               r"discover|father|mother|marr|rul|reign|govern|preside)\w*")
_DENIAL_TEMPLATES: tuple[tuple[str, str], ...] = (
    # "were not composed by V", "not choreographed to music composed by V" (no "but"/"by" in between: "not by X but by V" asserts V)
    ("not_by", _NOT + r"\s+(?:(?!by\b|but\b)[\w'-]+\s+){0,5}?by\s+(?:the\s+)?(?:\w+\s+){0,2}?{V}"),
    # "none of these ballets were composed by V"
    ("none_by", r"\b(?:none|neither|nothing)\b(?:(?!\bbut\b)[^.;])" + r"{0,40}?\bby\s+{V}"),
    # "not V", "wasn't V", "never V" (but not "not just V", which asserts, nor "isn't V's first role", a property)
    ("not_value", r"\b" + _NOT + r"\s+(?!(?:just|only|merely|simply)\b)(?:(?:even|actually|really|the|a|an)\s+)*{V}(?!['’]s\b)"),
    # "not the work of V"
    ("not_work_of", r"\b" + _NOT + r"\s+(?:the\s+)?(?:work|creation|composition|invention|brainchild|product)\s+of\s+{V}"),
    # "no historical evidence confirming that V composed", "nothing suggests V"
    ("no_evidence", r"\b(?:no|little|scant|without|lack of|lacks|lacking|absence of|nothing)\s+(?:\w+\s+){0,2}?(?:evidence|proof|record|records|"
    r"documentation|indication|basis|source|sources|suggests|indicates)\b[^.;]{0,60}?{V}"),
    # "mistakenly attributed to V", "misattributed to V"
    ("misattributed", r"\b(?:" + _ERR + r"\s+|mis)(?:attributed|credited|ascribed|assigned|identified|named|labeled|labelled)\s+(?:\w+\s+){0,3}?(?:to|as)\s+{V}"),
    # "V is incorrectly named", "V was never the composer", "V's authorship is disputed"
    ("value_is_wrong", r"{V}(?:'s)?\s+(?:\w+\s+){0,2}?(?:is|was|are|were|has been|have been|remains)\s+(?:often\s+|sometimes\s+|commonly\s+|widely\s+)?"
    r"(?:" + _ERR + r"|never|disputed|unfounded|false|incorrect|wrong|inaccurate|unsupported|doubtful|a myth|a misconception)\b"),
    # "V's authorship of the ballets is disputed", "V's involvement has been questioned"
    ("authorship_disputed", r"{V}'s\s+(?:authorship|involvement|role|claim|attribution|credit|contribution)\b[^.;]{0,40}?\b(?:is|was|are|were|remains|has been)\s+"
    r"(?:disputed|doubtful|unfounded|questioned|contested|rejected|false|incorrect|wrong|a myth|a misconception|unsupported)\b"),
    # "V is not the composer / the dog's name / the correct answer"
    ("value_is_not_role", r"{V}\s+(?:is|was|are|were)\s+not\s+(?:the\s+|a\s+|an\s+)?(?:\w+'s\s+)?(?:composer|author|creator|writer|director|voice|actor|"
    r"artist|singer|inventor|discoverer|founder|winner|name|answer|one|person|man|woman|correct|right|real|true|actual|original)\b"),
    # "V did not compose", "V never wrote", "V could not have created"
    ("value_did_not_verb", r"{V}\s+(?:did not|didn't|never|does not|doesn't|has not|hasn't|had not|hadn't|could not|couldn't|cannot|can't|would not|wouldn't)"
    r"\s+(?:\w+\s+){0,2}?" + _AUTH_VERBS),
    # "the myth that V composed", "the misconception that V ..."
    ("myth_that", r"\b(?:myth|misconception|rumou?r|legend|falsehood|false claim|false belief|error)\s+that\s+{V}"),
    # "supposedly composed by V", "the so-called V"
    ("supposedly", r"\b(?:supposedly|allegedly|purportedly|so-called)\s+(?:\w+\s+){0,3}?{V}"),
)
_denial_cache: dict[str, list[tuple[str, re.Pattern]]] = {}


# Компилирует шаблоны отрицания для одного паттерна значения (результат кэшируется).
# Инпут: re.Pattern алиасов значения
# Аутпут: list[tuple[str, re.Pattern]] имя формы отрицания и регулярка
def denial_patterns(value: re.Pattern) -> list[tuple[str, re.Pattern]]:
    key = value.pattern
    if key not in _denial_cache:
        v = "(?:" + value.pattern + ")"
        _denial_cache[key] = [(n, re.compile(t.replace("{V}", v), re.I)) for n, t in _DENIAL_TEMPLATES]
    return _denial_cache[key]


def denial_hits(text: str, pattern: re.Pattern) -> list[tuple[str, str, str]]:
    hits = []
    for sent in split_sentences(text):
        if not pattern.search(sent):
            continue
        for name, rx in denial_patterns(pattern):
            m = rx.search(sent)
            if m:
                hits.append((name, m.group(0), sent))
                break
    return hits


# Находит предложения, в которых значение упомянуто и в том же предложении отрицается ("not composed by X", "X did not write").
# Инпут: str текст; re.Pattern алиасов значения
# Аутпут: list[str] предложения с отрицанием (пустой список = все упоминания утвердительные)
def negated_mentions(text: str, pattern: re.Pattern) -> list[str]:
    return [sent for _, _, sent in denial_hits(text, pattern)]
