from __future__ import annotations

import json
import sys
from pathlib import Path

from ..signals.base import Context


def _import_cram(cram_repo: Path):
    repo = str(Path(cram_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from utils.re_weighting import Find_Best_Heads  # noqa: WPS433
    from utils.find_best_heads import casual_tracing_combine_all, find_top_k_heads  # noqa: WPS433
    return Find_Best_Heads, casual_tracing_combine_all, find_top_k_heads


# Готовит образцы для причинной трассировки CrAM: контексты ровно с одним ложным пассажем.
# Инпут: list[Context]; dict pair_id >> ложное значение
# Аутпут: list[dict] question, paras, fake_index, wrong_answer
def samples_from_contexts(contexts: list[Context], answer_false_by_pair: dict[str, str]) -> list[dict]:
    samples = []
    for ctx in contexts:
        false_idx = [i for i, d in enumerate(ctx.docs) if d.sealed.get("stance") == "false"]
        if len(false_idx) != 1:
            continue
        samples.append({"pair_id": ctx.pair_id, "question": ctx.question, "paras": ctx.texts(),
                        "fake_index": false_idx[0], "wrong_answer": answer_false_by_pair[ctx.pair_id]})
    return samples


# Прогоняет Find_Best_Heads CrAM: маскирует ложный пассаж в одной голове за раз и меряет падение логитов неверного ответа;
# возобновляемо, результаты дописываются в heads_scores.json.
# Инпут: str путь модели; путь к клону CrAM; list[dict] образцы; Path папка вывода
# Аутпут: Path heads_scores.json
def causal_tracing(model_path: str, cram_repo: str | Path, samples: list[dict], out_dir: Path) -> Path:
    Find_Best_Heads, _, _ = _import_cram(Path(cram_repo))
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "heads_scores.json"
    done: list = json.loads(scores_path.read_text()) if scores_path.exists() else []
    tracer = Find_Best_Heads(model_name=model_path)
    for idx, s in enumerate(samples):
        if idx < len(done):
            continue
        scores = [0 if i == s["fake_index"] else 1 for i in range(len(s["paras"]))]
        change = tracer.cal_logits(question=s["question"], paras=s["paras"], scores=scores,
                                   wrong_answer=s["wrong_answer"])
        done.append(change)
        scores_path.write_text(json.dumps(done), encoding="utf-8")
    return scores_path


# Усредняет эффекты по образцам и отбирает top-k голов кодом CrAM (по умолчанию все головы с положительным средним эффектом).
# Инпут: путь к клону CrAM; Path папка с heads_scores.json; int k или None
# Аутпут: dict число голов всего, положительных, k, путь selected_heads.json
def combine_and_select(cram_repo: str | Path, out_dir: Path, topk: int | None = None) -> dict:
    _, combine_all, select_topk = _import_cram(Path(cram_repo))
    ranked = combine_all(input_path=str(out_dir))          # writes heads_scores_mean.json
    n_positive = sum(1 for value, _ in ranked if value > 0)
    k = topk or n_positive
    select_topk(input_path=str(out_dir), topk=k)           # writes selected_heads.json
    return {"n_heads_total": len(ranked), "n_positive": n_positive, "topk": k,
            "selected_heads": str(out_dir / "selected_heads.json")}
