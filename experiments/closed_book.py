from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from witness_rag.cram.driver import CramReader  # noqa: E402
from witness_rag.eval.closed_book import run_closed_book  # noqa: E402
from witness_rag.io import data_path, load_config, read_jsonl, resolve  # noqa: E402
from witness_rag.schemas import Pair, from_dict  # noqa: E402


# Проба закрытой книги для ридера по всем парам, результат в data/closed_book/<reader>.jsonl.
# Инпут: --reader
# Аутпут: int код возврата
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reader", default="rd-llama3")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    pairs = [from_dict(Pair, r) for r in read_jsonl(data_path(cfg, "claims", "pairs.jsonl"))]
    reader = CramReader(cfg["readers"][args.reader], resolve(cfg["paths"]["cram_repo"]), None,
                        int(cfg["cram"]["max_new_tokens"]))
    out = data_path(cfg, "closed_book", f"{args.reader}.jsonl")
    run_closed_book(reader, args.reader, pairs, out)
    rows = read_jsonl(out)
    print(dict(collections.Counter(r["outcome"] for r in rows)), "->", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
