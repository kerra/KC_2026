from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable, Iterator


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Превращает относительный путь в абсолютный относительно корня репозитория (папка с configs/ и src/).
# Инпут: str | Path путь; Path корень (по умолчанию корень репозитория)
# Аутпут: Path абсолютный путь
def resolve(path_str: str | Path, root: Path | None = None) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else (root or repo_root()) / p


def read_yaml(path: str | Path) -> dict:
    import yaml  # local import: keeps the CPU path importable without yaml in tests that don't need it

    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Читает YAML-конфиг проекта и записывает корень репозитория в ключ _root.
# Инпут: str | Path путь к конфигу или None (configs/project.yaml)
# Аутпут: dict конфиг
def load_config(path: str | Path | None = None) -> dict:
    cfg = read_yaml(resolve(path or "configs/project.yaml"))
    cfg["_root"] = str(repo_root())
    return cfg


def data_path(cfg: dict, *parts: str) -> Path:
    return resolve(cfg["paths"]["data_dir"]).joinpath(*parts)


def results_path(cfg: dict, *parts: str) -> Path:
    return resolve(cfg["paths"]["results_dir"]).joinpath(*parts)


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


# Пишет объекты в jsonl по одному на строку, создавая папку при необходимости.
# Инпут: str | Path путь; Iterable[dict] строки; str режим открытия файла
# Аутпут: int число записанных строк
def write_jsonl(path: str | Path, rows: Iterable[dict], mode: str = "w") -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, mode, encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def append_jsonl(path: str | Path, row: dict) -> None:
    write_jsonl(path, [row], mode="a")


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Детерминированный 64-битный сид из произвольных частей через sha256; не зависит от PYTHONHASHSEED.
# Инпут: произвольные части (str, int, ...)
# Аутпут: int сид
def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def stable_rng(*parts: Any) -> random.Random:
    return random.Random(stable_seed(*parts))


def word_count(text: str) -> int:
    return len(text.split())
