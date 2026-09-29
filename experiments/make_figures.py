from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "witness_rag_mpl"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from claims import cluster_bootstrap, to_components  # noqa: E402
from witness_rag.io import data_path, load_config, read_json, read_jsonl, resolve  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
TRUE_C, FALSE_C, NEUTRAL_C, HIGHLIGHT, PANEL = "#2a78d6", "#e34948", "#c3c2b7", "#f9e2b0", "#f3f2ee"
COLOR = {"none": MUTED, "oracle": INK, "judge": "#4a3aa7", "shape": "#eb6834", "form": "#1baf7a", "form_lexicon": "#eda100",
         "dependence": "#e87ba4", "agreement": "#008300", "voting": "#2a78d6"}
MARKER = {"none": "o", "oracle": "D", "judge": "s", "shape": "^", "form": "o", "form_lexicon": "s", "dependence": "P",
          "agreement": "X", "voting": "v"}
LABEL = {"none": "none (plain RAG)", "oracle": "oracle (labels)", "judge": "judge (reader)", "form": "form (Binoculars)",
         "form_lexicon": "form (AI-marker lexicon)", "shape": "shape", "dependence": "dependence discount",
         "agreement": "agreement (discounted voting)", "voting": "voting"}
SCHEME_LABEL = {"none": "none", "sort_only": "sort by GPT score only", "gpt": "CrAM: GPT scores", "gpt_maxnorm": "GPT scores, soft, unsorted",
                "cnn": "rule: 'CNN news:' prefix", "shape": "shape score (truth-blind)", "ideal": "ideal (labels)"}
SCHEME_ORDER = ["none", "sort_only", "gpt", "gpt_maxnorm", "cnn", "shape", "ideal"]
SCHEME_COLOR = {"none": MUTED, "sort_only": MUTED, "gpt": COLOR["judge"], "gpt_maxnorm": COLOR["judge"], "cnn": INK2,
                "shape": COLOR["shape"], "ideal": INK}
CONDITIONS = [("clean", "clean retrieval", "no lie; machine-written text is true", "1.1, 2.1"),
              ("all_human", "all human-written", "authorship carries no information", "2.1"),
              ("all_ai_mixed", "all machine-written", "authorship carries no information", "1.1, 2.1"),
              ("ai_true_human_false", "machine true, human false", "machine-written means true", "2.1"),
              ("shortcut", "shortcut", "machine-written means false", "1.2, 2.1"),
              ("shortcut_cram", "CrAM's fakes", "machine-written and answer-shaped mean false", "1.1, 1.2"),
              ("shortcut_contrast", "contrastive lies", "the lie names the true value and denies it", "1.2"),
              ("independent", "independent lies", "k lies, each written separately", "2.2"),
              ("coordinated", "coordinated cluster", "k lies generated from one prompt", "2.2")]
CELL_ORDER = {"H_T": 0, "A_T": 1, "H_F": 2, "A_F": 3, "A_C": 4, "G_F": 5, "C_F": 6, "H_N": 7}
TRUTH = {"H_T": "T", "A_T": "T", "H_F": "F", "A_F": "F", "A_C": "F", "G_F": "F", "C_F": "F", "H_N": "N"}
TRUTH_COLOR = {"T": TRUE_C, "F": FALSE_C, "N": NEUTRAL_C}
AUTHOR = {"H_T": "H", "H_F": "H", "H_N": "H", "A_T": "A", "A_F": "A", "A_C": "Ac", "G_F": "G", "C_F": "C"}
ROLE = {"D2": "seed", "D3": "paraphrase", "D4": "copy"}
OUTCOME_COLOR = {"gold": TRUE_C, "attack": FALSE_C}


def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2, "axes.labelsize": 8.5,
        "axes.titlesize": 9.5, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 10, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "xtick.major.size": 0, "ytick.major.size": 0, "xtick.major.pad": 5, "ytick.major.pad": 6,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-",
        "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round", "lines.markersize": 7,
        "legend.frameon": False, "legend.fontsize": 8, "legend.handlelength": 1.4, "legend.borderaxespad": 0.3,
    })


def pct(rows, metric="gold") -> float:
    return 100 * sum(r["outcome"] == metric for r in rows) / len(rows) if rows else float("nan")


def sel(rows, signal, cond, k=None):
    conds = (cond,) if isinstance(cond, str) else tuple(cond)
    return [r for r in rows if r["signal"] == signal and r["condition"] in conds and (k is None or r["k"] == k)]


def finish(ax, grid="x") -> None:
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.7)
    ax.set_axisbelow(True)


def dot_ci(ax, y, est, lo, hi, color, marker="o", ms=7, lw=2.2, label=None, hollow=False, vertical=False) -> None:
    xs, ys = ([y, y], [lo, hi]) if vertical else ([lo, hi], [y, y])
    ax.plot(xs, ys, color=color, lw=lw, solid_capstyle="round", zorder=3)
    px, py = (y, est) if vertical else (est, y)
    ax.plot([px], [py], marker, ms=ms, color=color, mfc=SURFACE if hollow else color, mec=color if hollow else SURFACE,
            mew=1.6 if hollow else 1.3, zorder=4, label=label, ls="none")


def key(color, marker="o", label="", hollow=False, ms=7) -> Line2D:
    return Line2D([], [], marker=marker, ls="none", ms=ms, color=color, mfc=SURFACE if hollow else color,
                  mec=color if hollow else SURFACE, mew=1.6 if hollow else 1.0, label=label)


def save(fig, out: Path) -> None:
    fig.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# Раскладывает текст по словам внутри осей, переносит строки по ширине и подсвечивает фрагменты с номером-меткой в начале.
# Инпут: оси; str текст; list[(str фрагмент, str метка)]; float левый край, верх и ширина в долях осей; float кегль
# Аутпут: float нижняя граница набранного текста в долях осей
def flow_text(ax, text: str, spans, x0: float, top: float, width: float, size: float = 8.3, lead: float = 1.7) -> float:
    fig = ax.figure
    rend = fig.canvas.get_renderer()

    def w_of(s, sz, weight="normal"):
        t = ax.text(0, 0, s, fontsize=sz, fontweight=weight, transform=ax.transAxes)
        w = t.get_window_extent(rend).width / ax.bbox.width
        t.remove()
        return w

    tag_of = [None] * len(text)
    for frag, tag in spans:
        i = text.find(frag) if frag else -1
        if i >= 0:
            tag_of[i:i + len(frag)] = [tag] * len(frag)
    space, line = w_of("a a", size) - w_of("aa", size), size * lead / 72 * fig.dpi / ax.bbox.height
    asc, desc = size * 0.98 / 72 * fig.dpi / ax.bbox.height, size * 0.34 / 72 * fig.dpi / ax.bbox.height
    x, y, prev, placed = x0, top - line, None, []
    for m in re.finditer(r"\S+", text):
        tags = [t for t in tag_of[m.start():m.end()] if t]
        tag = tags[0] if tags else None
        head = tag is not None and tag != prev
        tw = w_of(tag, size - 1.8, "bold") + 0.004 if head else 0.0
        ww = w_of(m.group(), size)
        if x + tw + ww > x0 + width and x > x0:
            x, y = x0, y - line
        placed.append((x, x + tw + ww, y, tag, head, tw, m.group()))
        x += tw + ww + space
        prev = tag
    runs: list[list] = []
    for xl, xr, yy, tag, head, _, _ in placed:
        if tag and runs and runs[-1][3] == tag and runs[-1][2] == yy and not head:
            runs[-1][1] = xr
        elif tag:
            runs.append([xl, xr, yy, tag])
    for xl, xr, yy, _ in runs:
        ax.add_patch(Rectangle((xl - 0.003, yy - desc), xr - xl + 0.006, asc + desc, transform=ax.transAxes, fc=HIGHLIGHT, ec="none", zorder=1))
    for xl, _, yy, tag, head, tw, word in placed:
        if head:
            ax.text(xl + 0.001, yy + 0.3 * asc, tag, fontsize=size - 1.8, fontweight="bold", color=INK2, transform=ax.transAxes,
                    va="baseline", zorder=3)
        ax.text(xl + tw, yy, word, fontsize=size, color=INK, transform=ax.transAxes, va="baseline", zorder=3)
    return y - desc


# Рисунок 1: как устроены данные CrAM. Слева первая запись NQ-файла CrAM: её первый фейк с пронумерованными шаблонами
# и первый настоящий пассаж, с баллами GPT; справа доля фейков и настоящих пассажей с каждым шаблоном по обоим датасетам.
# Инпут: dict первая запись nq_1000_bge.json; dict critique.json; Path PNG
# Аутпут: None; файл записан
def fig_cram_fakes(rec: dict, critique: dict, out: Path) -> None:
    fig = plt.figure(figsize=(12.6, 4.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.34, left=0.02, right=0.98, top=0.86, bottom=0.22)
    ax = fig.add_subplot(gs[0])
    ax.set_axis_off()
    q, true, wrong = rec["question"], rec["reference"][0], rec["wrong answer"]
    ax.set_title("(a) The first record of CrAM's NQ file", loc="left", pad=20)
    ax.text(0, 1.02, f"question \"{q}\"    true value: {true}    CrAM's wrong value: {wrong}", fontsize=8, color=INK2,
            transform=ax.transAxes, va="bottom")
    first = lambda t, n=62: " ".join(t.split()[:n]) + " …"
    qw = q.lower().split()
    fake = first(rec["ori_fake"][0])
    restated = re.search(re.escape(qw[0]) + r".*?" + re.escape(qw[-1]), fake, flags=re.I)
    spans = [("CNN news:", "1"), (restated.group(0) if restated else "", "2"), ("rather than", "3"), (true, "4")]
    boxes = [(fake, spans, FALSE_C, f"CrAM's fake, first of three    GPT credibility score {rec['ori_fake_truthful_scores'][0]} / 10"),
             (first(rec["reranked_dense_ctxs"][0]), [(true, "4")], TRUE_C,
              f"retrieved passage, first of four    GPT credibility score {rec['reranked_dense_ctxs_truthful_scores'][0]} / 10")]
    top = 0.93
    for text, sp, rule, head in boxes:
        ax.text(0.02, top, head, fontsize=8.3, fontweight="bold", color=INK, transform=ax.transAxes, va="top")
        bottom = flow_text(ax, text, sp, 0.02, top - 0.035, 0.955)
        ax.add_patch(FancyBboxPatch((0.0, bottom - 0.03), 1.0, top - bottom + 0.06, boxstyle="round,pad=0,rounding_size=0.012",
                                    transform=ax.transAxes, fc=PANEL, ec="none", zorder=0))
        ax.add_patch(Rectangle((0.0, bottom - 0.03), 0.006, top - bottom + 0.06, transform=ax.transAxes, fc=rule, ec="none", zorder=0.5))
        top = bottom - 0.1

    ax = fig.add_subplot(gs[1])
    feats = [("cnn", "1  begins with \"CNN news:\""), ("restates_q", "2  restates the question"),
             ("contrast", "3  contrast construction\n    (\"rather than\", \"not X but Y\")"), ("mentions_true", "4  contains the true value")]
    ys = np.arange(len(feats))[::-1].astype(float)
    t = critique["T1_T2"]
    for (name, _), y in zip(feats, ys):
        for ds, marker, dy in (("nq", "o", 0.13), ("trivia", "s", -0.13)):
            a = t[ds]["audit"]
            fv, rv = 100 * a[f"fake_{name}"] / a["fakes"], 100 * a[f"real_{name}"] / a["reals"]
            ax.plot([rv, fv], [y + dy, y + dy], color=GRID, lw=2, zorder=2)
            ax.plot([rv], [y + dy], marker, color=TRUE_C, mec=SURFACE, mew=1.2, ms=7, zorder=3)
            ax.plot([fv], [y + dy], marker, color=FALSE_C, mec=SURFACE, mew=1.2, ms=7, zorder=3)
            ax.text(fv + 3, y + dy, f"{fv:.0f}%", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f[1] for f in feats], fontsize=8, color=INK)
    ax.set_xlim(-2, 104)
    ax.set_ylim(-0.6, len(feats) - 0.4)
    ax.set_xlabel("share of passages, %")
    ax.set_title("(b) Templates across all released passages", loc="left", pad=20)
    ax.text(0, 1.02, "3,000 fakes and 4,000 GPT-scored real passages per dataset", fontsize=8, color=INK2, transform=ax.transAxes, va="bottom")
    finish(ax, "x")
    ax.legend(handles=[key(FALSE_C, "o", "CrAM fake, NQ"), key(FALSE_C, "s", "CrAM fake, TriviaQA"),
                       key(TRUE_C, "o", "real passage, NQ"), key(TRUE_C, "s", "real passage, TriviaQA")],
              loc="upper center", bbox_to_anchor=(0.4, -0.16), ncol=2)
    save(fig, out)


# Рисунок 2: из чего состоят контексты: по семь пассажей, цвет = истинность, буква = автор; справа что проверяет
# условие и в каком клейме оно используется.
# Инпут: list[dict] строки test_v1; list[dict] строки контрастного прогона; dict doc_id >> строка лога; Path PNG
# Аутпут: None; файл записан
def fig_conditions(rows, contrast_rows, log, out: Path) -> None:
    first = {}
    for r in rows + contrast_rows:
        if r["signal"] == "none":
            first.setdefault((r["condition"], r["k"]), r)
    items = []
    for cond, name, tests, claims in CONDITIONS:
        ks = sorted(k for c, k in first if c == cond)
        for i, k in enumerate(ks):
            r = first[(cond, k)]
            cells = sorted(zip(r["cells"], r["doc_ids"]), key=lambda x: (CELL_ORDER[x[0]], log[x[1]]["dependence_level"] or "", x[1]))
            labs = [(c, AUTHOR[c] + {"D3": "p", "D4": "c"}.get(log[d]["dependence_level"], "") if c == "C_F" else AUTHOR[c]) for c, d in cells]
            note = "3 seeds, a paraphrase (Cp), a light-edit copy (Cc)" if cond == "coordinated" and k == 5 else (tests if i == 0 else "")
            items.append({"name": name if i == 0 else "", "k": k, "cells": labs, "note": note, "claims": claims if i == 0 else "", "group_start": i == 0})
    unit, left, right, top, bottom = 0.3, 2.35, 4.1, 0.62, 1.2
    n = len(items)
    W, H = left + 7 * unit + right, top + n * unit + bottom
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([left / W, bottom / H, 7 * unit / W, n * unit / H])
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_axis_off()
    for i, it in enumerate(items):
        if it["group_start"] and i:
            ax.plot([-8.2, 20.6], [i - 0.5, i - 0.5], color=GRID, lw=0.8, clip_on=False)
        for j, (cell, lab) in enumerate(it["cells"]):
            tr = TRUTH[cell]
            ax.add_patch(FancyBboxPatch((j - 0.4, i - 0.4), 0.8, 0.8, boxstyle="round,pad=0,rounding_size=0.14", fc=TRUTH_COLOR[tr], ec="none"))
            ax.text(j, i + 0.02, lab, ha="center", va="center", fontsize=7, fontweight="bold", color=INK if tr == "N" else "white")
        ax.text(-2.2, i, it["name"], ha="right", va="center", fontsize=8.3, fontweight="bold", color=INK)
        if it["k"]:
            ax.text(-0.75, i, f"k = {it['k']}", ha="right", va="center", fontsize=8, color=INK2)
        ax.text(7.1, i, it["note"], ha="left", va="center", fontsize=8, color=INK2)
        ax.text(18.6, i, it["claims"], ha="left", va="center", fontsize=8, color=INK)
    for x, lab, ha in ((3.0, "the seven passages of a context", "center"), (7.1, "what the condition varies", "left"), (18.6, "claims", "left")):
        ax.text(x, -1.05, lab, ha=ha, va="center", fontsize=8, fontweight="bold", color=INK)
    legend = [(TRUE_C, "true"), (FALSE_C, "false"), (NEUTRAL_C, "neutral filler")]
    for j, (c, lab) in enumerate(legend):
        x = -7.6 + j * 3.3
        ax.add_patch(FancyBboxPatch((x - 0.3, n + 0.35), 0.6, 0.6, boxstyle="round,pad=0,rounding_size=0.1", fc=c, ec="none", clip_on=False))
        ax.text(x + 0.55, n + 0.65, lab, va="center", fontsize=8, color=INK)
    ax.text(2.4, n + 0.65, "H human-written Wikipedia chunk    A machine-written (Qwen2.5, Llama-3.1, Aya)    "
            "Ac machine-written lie that names and denies the true value", va="center", fontsize=8, color=INK2)
    ax.text(2.4, n + 1.35, "G CrAM's released fake    C coordinated cluster: seed C, paraphrase Cp, light-edit copy Cc",
            va="center", fontsize=8, color=INK2)
    save(fig, out)


# Рисунок 3: как каждый арм распределяет внимание в одном вопросе: множитель каждого пассажа (высота столбика,
# цвет = истинность пассажа) и ответ ридера, в контексте с фейком CrAM и в скоординированном кластере из пяти лжей.
# Вопрос выбран правилом: первая тестовая пара, в которой ридер без взвешивания ошибается в обоих контекстах.
# Инпут: list[dict] строки test_v1; dict doc_id >> строка лога; dict pair_id >> пара; Path PNG
# Аутпут: None; файл записан
def fig_arms_example(rows, log, pairs, out: Path) -> None:
    arms = ["none", "oracle", "judge", "shape", "form", "form_lexicon", "dependence", "agreement", "voting"]
    contexts = [("shortcut_cram", 1, "(a) one CrAM fake against one human truth (CrAM's fakes, k = 1)"),
                ("coordinated", 5, "(b) a coordinated cluster of five lies (coordinated, k = 5)")]
    fooled = [{r["pair_id"] for r in rows if r["signal"] == "none" and r["condition"] == c and r["k"] == k and r["outcome"] == "attack"}
              for c, k, _ in contexts]
    pid = min(set.intersection(*fooled))
    p = pairs[pid]
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.4), sharey=True, gridspec_kw={"wspace": 0.3})
    fig.subplots_adjust(left=0.14, right=0.95, top=0.72, bottom=0.03)
    for ax, (cond, k, title) in zip(axes, contexts):
        by = {r["signal"]: r for r in rows if r["pair_id"] == pid and r["condition"] == cond and r["k"] == k}
        base = by["none"]
        docs = sorted(zip(base["cells"], base["doc_ids"], range(1, 8)),
                      key=lambda x: (CELL_ORDER[x[0]], log[x[1]]["dependence_level"] or "", x[1]))
        for i, arm in enumerate(arms):
            m = dict(zip(by[arm]["doc_ids"], by[arm]["multipliers"]))
            for j, (cell, d, _) in enumerate(docs):
                ax.add_patch(Rectangle((j - 0.2, i + 0.36), 0.4, -0.72, fc=GRID, ec="none", zorder=1))
                ax.add_patch(Rectangle((j - 0.2, i + 0.36), 0.4, -0.72 * m[d], fc=TRUTH_COLOR[TRUTH[cell]], ec="none", zorder=2))
            r = by[arm]
            ax.plot([7.1], [i], "o", ms=7, color=OUTCOME_COLOR.get(r["outcome"], MUTED), mec=SURFACE, zorder=3, clip_on=False)
            ans = r["answer"].strip().rstrip(".")
            ax.text(7.4, i, ans if len(ans) <= 16 else ans[:15] + "…", va="center", fontsize=8, color=INK, clip_on=False)
        for j, (cell, d, pos) in enumerate(docs):
            role = ROLE.get(log[d]["dependence_level"], "") if cell == "C_F" else {"G_F": "CrAM fake", "H_N": "neutral"}.get(cell, "true")
            ax.text(j, -0.75, f"{cell}\n{role}\n#{pos}", ha="center", va="bottom", fontsize=7, color=INK2, linespacing=1.15)
        ax.text(7.1, -0.75, "reader's\nanswer", ha="left", va="bottom", fontsize=7, color=INK2, linespacing=1.15)
        ax.set_xlim(-0.6, 6.6)
        ax.set_ylim(len(arms) - 0.5, -0.5)
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([LABEL[a] for a in arms], color=INK)
        ax.set_xticks([])
        ax.set_title(title, loc="left", pad=48)
    axes[1].tick_params(labelleft=False)
    fig.text(0.14, 0.975, f"Pair {pid}: \"{p['question']}\"    true value {p['answer_true']}, false value {p['answer_false']}",
             fontsize=9, fontweight="bold", color=INK)
    fig.text(0.14, 0.94, "bar = the attention multiplier the arm gives the passage (0.1 to 1), colour = truth of the passage; "
             "#n = position in the prompt; dot = the reader's answer is true (blue), false (red) or neither (gray)", fontsize=8, color=INK2)
    save(fig, out)


# Рисунок 4: четыре клейма под двумя вопросами, первичная статистика с 95% интервалом и интервалом с поправкой на четыре клейма.
# Инпут: dict claims.json; Path PNG
# Аутпут: None; файл записан
def fig_claims(claims: dict, out: Path) -> None:
    desc = {"1.1": "accuracy: shape score minus CrAM's GPT scores, on CrAM's protocol",
            "1.2": "flip rate: contrastive minus plain lies, unweighted reader",
            "2.1": "Δ accuracy: machine = false minus machine = true, two detectors",
            "2.2": "Δ accuracy vs none: agreement and voting, lie majority (k = 3, 5)"}
    cs = claims["claims"]
    fig, ax = plt.subplots(figsize=(11, 3.9))
    y, ticks, labels = 0.0, [], []
    for rq in ("1", "2"):
        ax.text(-0.02, y - 0.05, "\n".join(textwrap.wrap(f"RQ{rq}  {claims['rqs'][rq]}", 64)), transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=8.5, fontweight="bold", color=INK, linespacing=1.4)
        y -= 1
        for c in (c for c in cs if str(c["rq"]) == rq):
            p = c["primary"]
            ax.plot(p["intervals"]["claims"], [y, y], color=MUTED, lw=1.4, zorder=2)
            ax.plot(p["intervals"]["95"], [y, y], color=INK2, lw=4.5, solid_capstyle="round", zorder=3)
            ax.plot([p["estimate"]], [y], "o", ms=8, color=INK, mec=SURFACE, mew=1.5, zorder=4)
            iv = p["intervals"]["95"]
            ax.text(p["intervals"]["claims"][1] + 1.5, y, f"{p['estimate']:+.1f}  [{iv[0]:+.1f}, {iv[1]:+.1f}]", va="center", fontsize=8.3, color=INK)
            ticks.append(y)
            labels.append(f"{c['id']}  {c['short']} ({c['verdict']})\n{desc[c['id']]}")
            y -= 1
        y -= 0.4
    ax.axvline(0, color=AXIS, lw=1.1, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8, color=INK, linespacing=1.5)
    ax.set_ylim(y + 0.6, 0.5)
    ax.set_xlim(-32, 76)
    ax.set_xlabel("primary statistic, percentage points (thick: 95% interval; thin: α = 0.05/4)")
    finish(ax, "x")
    save(fig, out)


# Рисунок 5 (клейм 1.1): слева протокол CrAM, точность по схемам взвешивания при 1 и 3 фейках против потолка без фейка;
# справа наш тестовый сплит: сдвиг точности shape и судьи против none там, где форма совпадает с правдой, и там, где нет.
# Инпут: list[dict] строки nq_answers.jsonl; dict claims.json; Path PNG
# Аутпут: None; файл записан
def fig_cram_shape(protocol, claims: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 3.9), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.55})
    ax = axes[0]
    ys = np.arange(len(SCHEME_ORDER))[::-1]
    for s, y in zip(SCHEME_ORDER, ys):
        a1, a3 = (pct([r for r in protocol if r["scheme"] == s and r["fake_num"] == fn]) for fn in (1, 3))
        ax.plot([a3, a1], [y, y], color=GRID, lw=2.2, zorder=2)
        ax.plot([a1], [y], "o", ms=8, color=SCHEME_COLOR[s], mec=SURFACE, mew=1.3, zorder=3)
        ax.plot([a3], [y], "o", ms=7, color=SCHEME_COLOR[s], mfc=SURFACE, mew=1.6, zorder=3)
        if s in ("gpt", "shape", "none"):
            ax.text(max(a1, a3) + 1.2, y, f"{a1:.1f} / {a3:.1f}", va="center", fontsize=7.5, color=INK2)
    nf = pct([r for r in protocol if r["scheme"] == "no_fake"])
    ax.axvline(nf, color=INK2, lw=1.1, zorder=1)
    ax.text(nf, len(SCHEME_ORDER) - 0.35, f"no fake: {nf:.1f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([SCHEME_LABEL[s] for s in SCHEME_ORDER], color=INK)
    ax.set_xlim(0, 55)
    ax.set_ylim(-0.6, len(SCHEME_ORDER) - 0.1)
    ax.set_xlabel("accuracy, % of CrAM's 1,000 NQ questions")
    ax.set_title("(a) CrAM's protocol: their reader, heads, passages and fakes", loc="left", pad=14)
    ax.legend(handles=[key(INK2, "o", "1 fake + 4 retrieved passages"), key(INK2, "o", "3 fakes + 4 retrieved passages", hollow=True)],
              loc="lower left")
    finish(ax, "x")
    ax = axes[1]
    cells = [("shortcut_cram", "CrAM's fakes, k = 1 to 3\nshape tracks truth"), ("all_ai_mixed", "all machine-written\nshape balanced"),
             ("clean", "clean retrieval\nno lie")]
    split = next(c for c in claims["claims"] if c["id"] == "1.1")["supporting"]["test_split"]
    for i, (cond, _) in enumerate(cells):
        for arm, dy in (("judge", 0.14), ("shape", -0.14)):
            b = split[f"{arm}_{cond}"]
            dot_ci(ax, len(cells) - 1 - i + dy, b["estimate"], *b["intervals"]["95"], COLOR[arm], MARKER[arm],
                   label=LABEL[arm] if i == 0 else None)
    ax.axvline(0, color=AXIS, lw=1.1, zorder=1)
    ax.set_yticks(range(len(cells))[::-1])
    ax.set_yticklabels([c[1] for c in cells], color=INK, linespacing=1.4)
    ax.set_ylim(-0.6, len(cells) - 0.4)
    ax.set_xlabel("Δ accuracy vs none, points (95% CI)")
    ax.set_title("(b) Our test split, same reader", loc="left", pad=14)
    ax.legend(loc="lower right")
    finish(ax, "x")
    save(fig, out)


# Рисунок 6 (клейм 1.1): почему форма работает на бенчмарке CrAM: ROC на отложенной половине пассажей для балла GPT,
# четырёх шаблонных флагов и восьми признаков формы (NQ и TriviaQA) и аудит AUC формы без отдельных признаков и префикса.
# Инпут: dict critique.json; Path PNG
# Аутпут: None; файл записан
def fig_separability(critique: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.9), gridspec_kw={"width_ratios": [1, 1, 1.15], "wspace": 0.55})
    for ax, ds, title in zip(axes[:2], ("nq", "trivia"), ("(a) NQ", "(b) TriviaQA")):
        roc = critique["T1_T2"][ds]["roc_heldout"]
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1.1, zorder=1)
        for name, color, lab in (("gpt", COLOR["judge"], "GPT credibility score"), ("flags4", INK2, "4 template flags"),
                                 ("shape8", COLOR["shape"], "8 shape features")):
            pts = np.array(roc[name])
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2, drawstyle="steps-post" if name != "shape8" else "default", zorder=3,
                    label=f"{lab}, AUC {roc['auc_' + name]:.3f}".replace("AUC 0.", "AUC ."))
        fx, fy = roc["cnn_point"]
        ax.plot([fx], [fy], "D", ms=7, color=INK, mec=SURFACE, mew=1.2, zorder=4)
        ax.annotate("'CNN news:' prefix alone", (fx, fy), xytext=(0.12, fy - 0.05), fontsize=7.5, color=INK2, va="center",
                    arrowprops={"arrowstyle": "-", "color": AXIS, "lw": 0.8})
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("real passages flagged (false-positive rate)")
        ax.set_title(title + ": fake against real, held-out half", loc="left")
        ax.legend(loc="lower right", fontsize=7.5)
        finish(ax, "both")
    axes[0].set_ylabel("fakes flagged (true-positive rate)")
    ax = axes[2]
    a = critique["T1_audit"]
    tests = [("full", "all 8 features"), ("drop_one", "any 1 dropped (lowest)"), ("prefix_stripped", "'CNN news:' stripped"),
             ("random_100w_window", "random 100-word window"), ("first_sentence_dropped", "first sentence dropped")]
    ys = np.arange(len(tests))[::-1]
    for (name, _), y in zip(tests, ys):
        for ds, marker, dy in (("nq", "o", 0.12), ("trivia", "s", -0.12)):
            v = a[ds][name]
            v = min(v.values()) if isinstance(v, dict) else v
            ax.plot([v], [y + dy], marker, ms=7, color=COLOR["shape"], mec=SURFACE, mew=1.2, zorder=3)
            ax.text(v + 0.003, y + dy, f"{v:.3f}", va="center", fontsize=7.2, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([t[1] for t in tests], color=INK)
    ax.set_xlim(0.9, 1.012)
    ax.set_xlabel("AUC of the shape features, held-out half")
    ax.set_title("(c) What the shape separation rests on", loc="left")
    ax.legend(handles=[key(COLOR["shape"], "o", "NQ"), key(COLOR["shape"], "s", "TriviaQA")], loc="upper left")
    finish(ax, "x")
    save(fig, out)


# Рисунок 7 (клейм 1.2): доля флипов ридера без взвешивания при одной лжи против одной правды для четырёх видов лжи
# против того, насколько ложь похожа на ответ по баллу формы.
# Инпут: dict h3.json; Path PNG
# Аутпут: None; файл записан
def fig_lie_types(h3: dict, out: Path) -> None:
    kinds = [("plain lie, organic genre", MUTED, "o"), ("plain lie, targeted genre", MUTED, "s"),
             ("contrastive lie", FALSE_C, "D"), ("CrAM's fake", FALSE_C, "^")]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for name, color, marker in kinds:
        v = h3["lie_types"][name]
        x, (f, lo, hi, n) = v["mean_p_targeted"], v["flip"]
        dot_ci(ax, x, f, lo, hi, color, marker, ms=8, vertical=True)
        left = name == "plain lie, organic genre"
        ax.text(x - 0.025 if left else x + 0.025, f, f"{name}\n{f:.0f}% of {n} contexts", va="center", ha="right" if left else "left",
                fontsize=7.8, color=INK, linespacing=1.3)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(0, 72)
    ax.set_xlabel("how answer-shaped the lie reads: mean p(targeted) of the shape score")
    ax.set_ylabel("flip rate of the unweighted reader, % (95% CI)")
    ax.set_title("One lie against one human truth (k = 1)", loc="left")
    ax.legend(handles=[key(FALSE_C, "D", "names the true value and denies it"), key(MUTED, "o", "states the wrong value only")],
              loc="upper left")
    finish(ax, "both")
    save(fig, out)


# Рисунок 8 (клейм 1.2): флипы при простых и контрастных ложных пассажах по армам на общих парах при k = 1 и 3.
# Инпут: dict h3.json; Path PNG
# Аутпут: None; файл записан
def fig_h3(h3: dict, out: Path) -> None:
    arms = ["none", "shape", "form_lexicon", "judge", "oracle"]
    by = {(r["k"], r["arm"]): r for r in h3["by_arm"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, k in zip(axes, (1, 3)):
        ys = np.arange(len(arms))[::-1]
        for a, y in zip(arms, ys):
            r = by[(k, a)]
            ax.plot([r["plain_flip"], r["contrast_flip"]], [y, y], color=GRID, lw=2.4, zorder=2)
            ax.plot([r["plain_flip"]], [y], "o", ms=8, color=MUTED, mec=SURFACE, mew=1.3, zorder=3)
            ax.plot([r["contrast_flip"]], [y], "D", ms=7.5, color=FALSE_C, mec=SURFACE, mew=1.3, zorder=3)
            ax.text(106, y, f"{r['delta_flip']:+.0f}  [{r['lo']:+.0f}, {r['hi']:+.0f}]", va="center", fontsize=7.8, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([LABEL[a] for a in arms], color=INK)
        ax.set_xlim(-2, 130)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel("flip rate, % (the answer contains the wrong value)")
        ax.set_title(f"k = {k}: {by[(k, 'none')]['n']} test pairs with both contexts", loc="left")
        ax.text(106, len(arms) - 0.45, "Δ [95% CI]", fontsize=7.8, color=INK2, va="bottom")
        finish(ax, "x")
    fig.legend(handles=[key(MUTED, "o", "plain false passage (A_F)"), key(FALSE_C, "D", "contrastive false passage (A_C: 'X, not Y')")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1))
    save(fig, out)


# Рисунок 9: вернёт ли ридер правду, названную внутри фейка CrAM: точность с одним фейком минус точность без фейка
# по группам вопросов (истина в найденных пассажах, нет её там, есть только в фейке) и схемам взвешивания.
# Инпут: dict critique.json; Path PNG
# Аутпут: None; файл записан
def fig_truth_in_context(critique: dict, out: Path) -> None:
    c = critique["containment"]
    groups = [("present", f"true value in a retrieved passage\n{c['present']} questions; no fake: {c['accuracy']['present']['no_fake']:.1f}%"),
              ("absent", f"true value in no retrieved passage\n{c['absent']} questions; no fake: {c['accuracy']['absent']['no_fake']:.1f}%"),
              ("only_in_fake", f"of these, named only by the fake\n{c['truth_only_in_fake']} questions; no fake: "
                               f"{c['accuracy']['only_in_fake']['no_fake']:.1f}%")]
    schemes = [("none", COLOR["none"], "o", "none"), ("gpt", COLOR["judge"], "s", "CrAM: GPT scores"),
               ("shape", COLOR["shape"], "^", "shape score"), ("ideal", INK, "D", "ideal (labels)")]
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    for g, (grp, _) in enumerate(groups):
        for s, (name, color, marker, lab) in enumerate(schemes):
            est, lo, hi, _ = c["vs_no_fake"][f"{grp}/{name}"]
            dot_ci(ax, len(groups) - 1 - g + 0.3 - 0.2 * s, est, lo, hi, color, marker, label=lab if g == 0 else None)
    ax.axvline(0, color=INK2, lw=1.1, zorder=1)
    ax.text(0.5, len(groups) - 0.42, "no fake", fontsize=7.5, color=INK2, va="bottom")
    ax.set_yticks(range(len(groups))[::-1])
    ax.set_yticklabels([g[1] for g in groups], color=INK, linespacing=1.5)
    ax.set_ylim(-0.6, len(groups) - 0.4)
    ax.set_xlabel("accuracy with one CrAM fake minus accuracy with no fake, points (95% CI, paired by question)")
    ax.set_title("CrAM's NQ protocol, one fake, grouped by where the true value appears", loc="left")
    ax.legend(loc="lower left", ncol=2)
    finish(ax, "x")
    save(fig, out)


# Рисунок 10 (клейм 2.1): слева что видят детекторы авторства: p(машинный текст) каждого тестового пассажа по ячейкам,
# справа сдвиг точности двух армов авторства против none по контекстам, упорядоченным по тому, что значит машинный текст.
# Инпут: dict claims.json; Path PNG
# Аутпут: None; файл записан
def fig_authorship(claims: dict, out: Path) -> None:
    det = next(c for c in claims["claims"] if c["id"] == "2.1")["supporting"]["detector_scores"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.1), gridspec_kw={"width_ratios": [0.85, 1.25], "wspace": 0.22})
    ax = axes[0]
    cells = ["H_T", "H_F", "A_T", "A_F"]
    rng = np.random.default_rng(0)
    for arm, dx in (("form", -0.17), ("form_lexicon", 0.17)):
        for i, c in enumerate(cells):
            v = np.array(det[arm]["by_cell"][c])
            ax.scatter(i + dx + rng.uniform(-0.08, 0.08, len(v)), v, s=6, color=COLOR[arm], alpha=0.45, lw=0, zorder=2)
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            ax.plot([i + dx, i + dx], [q1, q3], color=INK, lw=2, zorder=3)
            ax.plot([i + dx - 0.1, i + dx + 0.1], [med, med], color=INK, lw=2.4, zorder=3)
    ax.axvline(1.5, color=AXIS, lw=1, zorder=1)
    ax.text(0.5, 1.07, "human-written", ha="center", fontsize=8, color=INK2)
    ax.text(2.5, 1.07, "machine-written", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels(["true (H_T)", "false (H_F)", "true (A_T)", "false (A_F)"], color=INK)
    ax.set_ylim(-0.03, 1.13)
    ax.set_ylabel("calibrated p(machine-written)")
    ax.set_title("(a) What the detectors see: test-split passages", loc="left", pad=16)
    f, fl = det["form"], det["form_lexicon"]
    ax.text(0.01, -0.2, f"AUC machine vs human: {f['auc_machine_vs_human']:.2f} (Binoculars), {fl['auc_machine_vs_human']:.2f} (lexicon); "
            f"false vs true within one author: {min(f['auc_false_vs_true_human'], f['auc_false_vs_true_machine'], fl['auc_false_vs_true_human'], fl['auc_false_vs_true_machine']):.2f}"
            f" to {max(f['auc_false_vs_true_human'], f['auc_false_vs_true_machine'], fl['auc_false_vs_true_human'], fl['auc_false_vs_true_machine']):.2f}",
            transform=ax.transAxes, fontsize=7.6, color=INK2)
    ax.legend(handles=[key(COLOR["form"], "o", LABEL["form"], ms=6), key(COLOR["form_lexicon"], "o", LABEL["form_lexicon"], ms=6),
                       Line2D([], [], color=INK, lw=2, label="median and interquartile range")], loc="lower left", fontsize=7.4)
    finish(ax, "y")
    ax = axes[1]
    ctx = [("shortcut", "shortcut, k = 1 to 3\nH_T + k A_F"), ("all_human", "all human\nH_T + H_F"),
           ("all_ai_mixed", "all machine\n2 A_T + 2 A_F"), ("ai_true_human_false", "machine true,\nhuman false\n2 A_T + H_F"),
           ("clean", "clean\nH_T + 2 A_T")]
    bands = [(0, 0, "machine-written\nmeans false"), (1, 2, "authorship\nuninformative"), (3, 4, "machine-written\nmeans true")]
    by_context = next(c for c in claims["claims"] if c["id"] == "2.1")["supporting"]["by_context"]
    for i, (cond, _) in enumerate(ctx):
        for arm, dx in (("form", -0.12), ("form_lexicon", 0.12)):
            b = by_context[f"{arm}_{cond}"]
            dot_ci(ax, i + dx, b["estimate"], *b["intervals"]["95"], COLOR[arm], MARKER[arm], vertical=True,
                   label=LABEL[arm] if i == 0 else None)
    for lo_i, hi_i, name in bands:
        ax.text((lo_i + hi_i) / 2, 32, name, ha="center", va="center", fontsize=8, color=INK2, linespacing=1.3)
        if lo_i:
            ax.axvline(lo_i - 0.5, color=AXIS, lw=1, zorder=1)
    ax.axhline(0, color=INK2, lw=1.1, zorder=1)
    ax.set_xticks(range(len(ctx)))
    ax.set_xticklabels([c[1] for c in ctx], color=INK, linespacing=1.3)
    ax.set_xlim(-0.5, len(ctx) - 0.5)
    ax.set_ylim(-58, 40)
    ax.set_ylabel("Δ accuracy vs none, points (95% CI)")
    ax.set_title("(b) What weighting by authorship does to accuracy", loc="left", pad=16)
    ax.legend(loc="lower left")
    finish(ax, "y")
    save(fig, out)


# Рисунок 11 (клейм 2.2): точность армов подсчёта свидетелей, дисконта, судьи и оракула по числу лжей k в условиях
# coordinated и independent, и 3-граммный Жаккар внутри кластера против порога слияния дисконта.
# Инпут: list[dict] строки test_v1; dict claims.json; float порог Жаккара; Path PNG
# Аутпут: None; файл записан
def fig_counting(rows, claims: dict, threshold: float, out: Path) -> None:
    arms = ["none", "oracle", "judge", "dependence", "agreement", "voting"]
    sup = next(c for c in claims["claims"] if c["id"] == "2.2")["supporting"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.1), gridspec_kw={"width_ratios": [1, 0.8, 1.05], "wspace": 0.42})
    for ax, cond, title in zip(axes[:2], ("coordinated", "independent"),
                               ("(a) Coordinated: k lies from one prompt", "(b) Independent: k lies written separately")):
        ks = sorted({r["k"] for r in rows if r["condition"] == cond})
        for arm in arms:
            ax.plot(ks, [pct(sel(rows, arm, cond, k)) for k in ks], marker=MARKER[arm], color=COLOR[arm], lw=2, ms=7, mec=SURFACE,
                    mew=1.2, label=LABEL[arm], zorder=3)
        ax.set_xticks(ks)
        ax.set_xlabel("k false passages (the true side has 2)")
        ax.set_title(title, loc="left")
        ax.set_ylim(0, 100)
        finish(ax, "y")
    axes[0].set_ylabel("accuracy, % (the answer contains the true value)")
    axes[1].set_yticklabels([])
    axes[1].legend(loc="lower left", fontsize=7.4)
    ax = axes[2]
    jac = sup["jaccard"]
    merges = sup["merges"]["5"]
    names = [("seed and its copy", f"seed and its\nlight-edit copy\nmerged {merges['D4_merged']} of {merges['D4_present']}"),
             ("seed and its paraphrase", f"seed and its\nparaphrase\nmerged {merges['D3_merged']} of {merges['D3_present']}"),
             ("two seeds", f"two seeds\nof one cluster\nmerged {merges['seed_pairs_merged']} of {merges['seed_pairs']}"),
             ("true passage and a seed", "true passage\nand a seed")]
    rng = np.random.default_rng(0)
    ys = np.arange(len(names))[::-1]
    for (name, _), y in zip(names, ys):
        v = np.array(jac[name]["values"])
        ax.scatter(v, y + rng.uniform(-0.17, 0.17, len(v)), s=9, color=COLOR["dependence"], alpha=0.55, lw=0, zorder=2)
        ax.plot([jac[name]["median"]] * 2, [y - 0.26, y + 0.26], color=INK, lw=2.4, zorder=3)
    ax.axvline(threshold, color=INK2, lw=1.1, zorder=1)
    ax.text(threshold + 0.01, len(names) - 0.45, f"merge threshold {threshold:.2f}", fontsize=7.5, color=INK2, va="bottom")
    ax.set_yticks(ys)
    ax.set_yticklabels([n[1] for n in names], color=INK, linespacing=1.3)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(-0.6, len(names) - 0.2)
    ax.set_xlabel("3-gram Jaccard similarity (bar = median)")
    ax.set_title("(c) What the discount sees in a test cluster", loc="left")
    finish(ax, "x")
    save(fig, out)


# Рисунок 12: сводка всех армов по всем условиям тестового сплита: сдвиг точности против none (синий = выигрыш,
# красный = потеря), сверху точность none.
# Инпут: list[dict] строки test_v1; Path PNG
# Аутпут: None; файл записан
def fig_scorecard(rows, out: Path) -> None:
    arms = ["oracle", "judge", "shape", "form", "form_lexicon", "dependence", "agreement", "voting"]
    cols = [(c, k, name) for c, name, _, _ in CONDITIONS if c != "shortcut_contrast"
            for k in sorted({r["k"] for r in rows if r["condition"] == c})]
    none = np.array([pct(sel(rows, "none", c, k)) for c, k, _ in cols])
    grid = np.array([[pct(sel(rows, a, c, k)) for c, k, _ in cols] for a in arms]) - none
    cmap = LinearSegmentedColormap.from_list("div", [FALSE_C, "#f0efec", TRUE_C])
    fig, ax = plt.subplots(figsize=(13, 4.3))
    im = ax.imshow(grid, cmap=cmap, vmin=-50, vmax=50, aspect="auto")
    for i in range(len(arms)):
        for j in range(len(cols)):
            v = grid[i, j]
            ax.text(j, i, f"{v:+.0f}" if round(v) else "0", ha="center", va="center", fontsize=7.4, color="white" if abs(v) >= 30 else INK)
    for j, v in enumerate(none):
        ax.text(j, -0.85, f"{v:.0f}", ha="center", va="center", fontsize=7.4, color=INK2)
    ax.text(-0.6, -0.85, "none, accuracy %", ha="right", va="center", fontsize=7.6, color=INK2)
    single = {"clean": "clean\nretrieval", "all_human": "all\nhuman", "all_ai_mixed": "all\nmachine", "ai_true_human_false": "machine true,\nhuman false"}
    starts = [j for j, (c, _, _) in enumerate(cols) if j == 0 or cols[j - 1][0] != c]
    for j in starts:
        c, _, name = cols[j]
        span = sum(1 for x in cols if x[0] == c)
        if span > 1:
            ax.text(j + (span - 1) / 2, -1.75, name, ha="center", va="center", fontsize=7.8, fontweight="bold", color=INK)
        if j:
            ax.axvline(j - 0.5, color=SURFACE, lw=3)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"k = {k}" if k else single[c] for c, k, _ in cols], fontsize=7.4, linespacing=1.2)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([LABEL[a] for a in arms], color=INK)
    ax.set_ylim(len(arms) - 0.5, -2.2)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.2, aspect=50, shrink=0.7)
    cb.set_label("Δ accuracy vs none, points", color=INK2, fontsize=8)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.4, colors=INK2, length=0)
    save(fig, out)


# Таблица 1: четыре клейма с первичной статистикой, интервалами, данными и вердиктом.
# Инпут: dict claims.json
# Аутпут: str markdown
def table_claims(claims: dict) -> str:
    lines = ["| claim | primary statistic | estimate, 95% CI | α = 0.05/4 | data | verdict |", "|---|---|---|---|---|---|"]
    for c in claims["claims"]:
        p, iv = c["primary"], c["primary"]["intervals"]
        lines.append(f"| {c['id']} {c['short']} | {c['statistic']} | {p['estimate']:+.1f} [{iv['95'][0]:+.1f}, {iv['95'][1]:+.1f}] | "
                     f"[{iv['claims'][0]:+.1f}, {iv['claims'][1]:+.1f}] | {c['data']} | **{c['verdict']}** |")
    strict = all(c["holds"]["claims"] for c in claims["claims"])
    lines += ["", f"Cluster bootstrap over the repeating unit (question pair, or CrAM question), {claims['n_resamples']:,} resamples, seed "
              f"{claims['seed']}. All four intervals {'also' if strict else 'do not all'} exclude zero at α = 0.05/4, the correction for four claims."]
    return "\n".join(lines) + "\n"


# Таблица 2: протокол CrAM по схемам взвешивания: точность, флипы и парная по вопросам разность против схемы none.
# Инпут: list[dict] строки nq_answers.jsonl; int ресэмплов; int сид
# Аутпут: str markdown
def table_cram_protocol(rows, n_boot: int, seed: int) -> str:
    idx = {(r["id"], r["fake_num"], r["scheme"]): r["outcome"] for r in rows}
    ids = sorted({r["id"] for r in rows})
    lines = ["| fakes | weighting scheme | n | accuracy | flip | Δ accuracy vs none [95% CI] |", "|---|---|---|---|---|---|"]
    for fn in (1, 3):
        for s in SCHEME_ORDER:
            g = [r for r in rows if r["scheme"] == s and r["fake_num"] == fn]
            if not g:
                continue
            delta = ""
            if s != "none":
                d = {str(q): [float(idx[(q, fn, s)] == "gold") - float(idx[(q, fn, "none")] == "gold")] for q in ids
                     if (q, fn, s) in idx and (q, fn, "none") in idx}
                b = cluster_bootstrap(to_components({"d": d}, [str(q) for q in ids]), lambda m: m["d"], n_boot, seed)
                lo, hi = b["intervals"]["95"]
                delta = f"{100 * b['estimate']:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            lines.append(f"| {fn} | {SCHEME_LABEL[s]} | {len(g)} | {pct(g):.1f} | {pct(g, 'attack'):.1f} | {delta} |")
    nf = [r for r in rows if r["scheme"] == "no_fake"]
    if nf:
        lines.append(f"| 0 | no fake (the 4 retrieved passages only) | {len(nf)} | {pct(nf):.1f} | {pct(nf, 'attack'):.1f} | |")
    lines += ["", "CrAM's 1,000 NQ questions, their 4 reranked passages plus their fakes, their 894 selected Llama-3 heads, reader "
              "Meta-Llama-3-8B-Instruct; outcome = the answer contains the reference (or an alias) / the wrong answer. Δ paired by question, "
              f"percentile bootstrap ({n_boot:,} resamples, seed {seed})."]
    return "\n".join(lines) + "\n"


# Таблица 3: что отделяет фейки CrAM от настоящих пассажей и как устроены выпущенные фейки (T1, T2, аудит AUC формы).
# Инпут: dict critique.json
# Аутпут: str markdown
def table_critique(critique: dict) -> str:
    t, a = critique["T1_T2"], critique["T1_audit"]
    lines = ["| | NQ | TriviaQA |", "|---|---|---|"]
    row = lambda name, f: lines.append(f"| {name} | {f(t['nq'])} | {f(t['trivia'])} |")
    row("released fakes / real passages scored by gpt-3.5", lambda v: f"{v['n_fakes']:,} / {v['n_reals']:,}")
    row("GPT credibility score, mean fake / real", lambda v: f"{v['gpt_score_mean_fake']} / {v['gpt_score_mean_real']}")
    row("AUC fake vs real: GPT score", lambda v: f"{v['auc_gpt_score_detects_fake']:.2f}")
    row("AUC: 'CNN news:' prefix alone", lambda v: f"{v['auc_cnn_prefix_alone']:.2f}")
    row("AUC: 4 template flags (held-out half)", lambda v: f"{v['auc_template4_detects_fake_heldout']:.2f}")
    row("AUC: 8 shape features (held-out half)", lambda v: f"{v['auc_shape8_detects_fake_heldout']:.3f}")
    row("R² of the GPT score from the 4 flags", lambda v: f"{v['r2_gpt_score_from_templates']:.2f}")
    row("fakes beginning with 'CNN news:'", lambda v: f"{100 * v['audit_shares']['fake_cnn']:.0f}%")
    row("fakes with a contrast construction", lambda v: f"{100 * v['audit_shares']['fake_contrast']:.0f}%")
    row("fakes restating the question", lambda v: f"{100 * v['audit_shares']['fake_restates_q']:.0f}%")
    row("fakes naming the true value they deny", lambda v: f"{100 * v['audit_shares']['fake_mentions_true']:.0f}%")
    row("fakes not asserting the wrong value or affirming the true one", lambda v: f"{100 * v['audit_shares']['fake_not_asserting_wrong_or_affirming_true']:.1f}%")
    arow = lambda name, f: lines.append(f"| {name} | {f(a['nq'])} | {f(a['trivia'])} |")
    arow("shape AUC, lowest after dropping one feature", lambda v: f"{min(v['drop_one'].values()):.3f}")
    arow("shape AUC, 'CNN news:' prefix stripped", lambda v: f"{v['prefix_stripped']:.3f}")
    arow("shape AUC, random 100-word window", lambda v: f"{v['random_100w_window']:.3f}")
    arow("shape AUC, opening sentence dropped", lambda v: f"{v['first_sentence_dropped']:.3f}")
    lines += ["", "AUCs of fitted classifiers are measured on the held-out half of the scored passages. The audit rows show that no single "
              "feature or string carries the separation."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="test_v1")
    ap.add_argument("--figures", default="figures")
    ap.add_argument("--tables", default="tables")
    args = ap.parse_args(argv)
    cfg = load_config()
    setup_style()
    fig_dir, tab_dir = resolve(args.figures), resolve(args.tables)
    fig_dir.mkdir(parents=True, exist_ok=True); tab_dir.mkdir(parents=True, exist_ok=True)
    for old in list(fig_dir.glob("*.png")) + list(tab_dir.glob("*.md")):
        old.unlink()

    rows = read_jsonl(resolve(f"results/{args.run_id}/answers.jsonl"))
    contrast_rows = read_jsonl(resolve("results/h3_contrast/answers.jsonl"))
    claims = read_json(resolve("results/claims/claims.json"))
    h3 = read_json(resolve("results/h3_contrast/h3.json"))
    critique = read_json(resolve("results/cram_critique/critique.json"))
    protocol = read_jsonl(resolve("results/cram_protocol/nq_answers.jsonl"))
    log = {r["doc_id"]: r for r in read_jsonl(data_path(cfg, "production_log.jsonl"))}
    pairs = {p["pair_id"]: p for p in read_jsonl(data_path(cfg, "claims", "pairs.jsonl"))}
    cram_first = read_json(resolve(cfg["paths"]["cram_repo"]) / "nq_1000_bge.json")[0]

    fig_cram_fakes(cram_first, critique, fig_dir / "fig01_cram_fakes.png")
    fig_conditions(rows, contrast_rows, log, fig_dir / "fig02_conditions.png")
    fig_arms_example(rows, log, pairs, fig_dir / "fig03_arms_example.png")
    fig_claims(claims, fig_dir / "fig04_claims.png")
    fig_cram_shape(protocol, claims, fig_dir / "fig05_cram_shape.png")
    fig_separability(critique, fig_dir / "fig06_separability.png")
    fig_lie_types(h3, fig_dir / "fig07_lie_types.png")
    fig_h3(h3, fig_dir / "fig08_h3_contrast.png")
    fig_truth_in_context(critique, fig_dir / "fig09_truth_in_context.png")
    fig_authorship(claims, fig_dir / "fig10_authorship.png")
    fig_counting(rows, claims, cfg["signals"]["dependence"]["jaccard_threshold"], fig_dir / "fig11_counting.png")
    fig_scorecard(rows, fig_dir / "fig12_scorecard.png")

    (tab_dir / "table1_claims.md").write_text(table_claims(claims), encoding="utf-8")
    (tab_dir / "table2_cram_protocol.md").write_text(table_cram_protocol(protocol, claims["n_resamples"], claims["seed"]), encoding="utf-8")
    (tab_dir / "table3_cram_critique.md").write_text(table_critique(critique), encoding="utf-8")
    print("figures:", sorted(p.name for p in fig_dir.glob("*.png")))
    print("tables:", sorted(p.name for p in tab_dir.glob("*.md")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
