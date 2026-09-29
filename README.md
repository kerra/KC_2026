# Credibility Without a Judge

## Running the code

CrAM is cloned separately, next to this repository; its files supply the questions and fakes, and its hook is imported unchanged. Steps 2 to 8 need one 24 GB GPU (about 45 hours on an RTX 3090) and Hugging Face access to the gated Llama-3, Llama-3.1 and Aya Expanse models.

```bash
git clone https://github.com/Aatrox103/CrAM ../CrAM      # from the root of this repository
pip install -r requirements.txt                          # CPU: steps 1 and 9, tests
pip install -r requirements-gpu.txt                      # GPU: steps 2 to 8 (transformers 5.17.0)
bash scripts/1_build_pairs.sh                            # pairs, human passages, generation plan
bash scripts/2_generate_corpus.sh                        # generation, sealing, QA
SEED_OFFSET=100000 bash scripts/3_repair_round.sh        # then 200000 and 300000 with REASSIGN=1, until QA is clean
bash scripts/4_reader_setup.sh                           # hook check, calibrators, closed-book probe, heads, judge check
bash scripts/5_dev_run.sh                                # all arms, dev split
bash scripts/6_test_run.sh                               # all arms, test split
bash scripts/7_cram_protocol.sh                          # CrAM's protocol under alternative weights
bash scripts/8_contrastive_cell.sh                       # contrastive lies and their run
bash scripts/9_analyses.sh                               # the four claims, figures, tables
PYTHONPATH=src python3 -m pytest -q                      # 60 CPU tests
```

Only `data/claims/` ships with the code: the 172 question pairs taken from CrAM's files (150 active after the repair rounds excluded 22) and the dev/test split. With them step 1 rebuilds the reported pairs, split and human passages exactly; the generated passages come from steps 2 and 3. The rest of `data/`, and `results/`, `figures/` and `tables/`, is written by the scripts and ignored by git. The hand-made inputs live in `configs/`: generation prompts, marker lexicon, thresholds, conditions.
