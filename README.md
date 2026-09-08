# TRACE — Artifact Release

TRACE recasts inline-aware binary code similarity detection from whole-function
matching to semantic-unit retrieval: each function is decomposed into
basic-block semantic units, encoded into a set of semantic representations, and
matched through a hierarchical retrieval framework (IVF candidate generation →
multi-channel reranking → verification).

This artifact reproduces the paper's RQ1–RQ4 from released embeddings and
materialized pools. **The encoder (base and fine-tuned) is not distributed**
per our group's policy; all embeddings the encoder would produce are shipped as
finished artifacts, and every retrieval / reranking / verification /
evaluation component is released as runnable code.

## Layout

```
src/
  extraction/   bb_slice_extractor.py       semantic-unit extraction (runnable on data/cve75)
  retrieval/    pool_inline_search.py + pool/index builders
  verify/       symbolic-anchor verifier + LLM verifier
  baselines/    DEJINA global-ANN / HermesSim / CI-Detector adapters
  ablation/     pool_inline_search_ablation.py  (RQ3: A1/A2/A3 in one run)
  bench/        efficiency benchmarks
data/cve75/     the inlined-CVE dataset: stripped query binaries, standalone GT
                binaries, decompiled features, and cases.json provenance
artifacts/      (not in the code repo — download from Hugging Face; see below)
  doi1/         RQ1/2/3 query pkls + materialized 10K/100K pools (XO / XA / XM)
  doi2/         RQ4: 75 CVE query pkls + 1M-pool runtime + minimal pool dir
  faiss/        the shared IVF bucket index (nlist=262144, trained on the pool corpus)
```

## Getting the data and artifacts

The code repo carries only `src/`, `data/cve75/cases.json`, and this README.
The datasets and the multi-vector pools/embeddings are hosted on Hugging Face:

> **https://huggingface.co/datasets/ICSE2027/TRACE**

Four bundles are published there. Download only what a given RQ needs:

| bundle | size | contents | needed for |
| --- | --- | --- | --- |
| `TRACE-code.tar.gz` | 125 KB | this code tree (same as the repo) | — |
| `TRACE-data.tar.gz` | 156 MB | `data/cve75/` (the inlined-CVE dataset) | dataset / verifiers |
| `TRACE-artifacts-inline.tar` | 34 GB | `artifacts/doi1/` + `artifacts/faiss/` | RQ1, RQ2, RQ3 |
| `TRACE-artifacts-cve.tar` | 31 GB | `artifacts/doi2/` (1M CVE pool runtime) | RQ4 |

Download with the `huggingface_hub` CLI (`pip install -U huggingface_hub`), then
unpack **from the repository root** so the paths land in place:

```bash
REPO=ICSE2027/TRACE
# dataset (needed by the anchor / LLM verifiers)
huggingface-cli download $REPO TRACE-data.tar.gz          --repo-type dataset --local-dir .
tar xf TRACE-data.tar.gz            # -> data/cve75/

# RQ1 / RQ2 / RQ3 artifacts
huggingface-cli download $REPO TRACE-artifacts-inline.tar --repo-type dataset --local-dir .
tar xf TRACE-artifacts-inline.tar   # -> artifacts/doi1/ , artifacts/faiss/

# RQ4 artifacts
huggingface-cli download $REPO TRACE-artifacts-cve.tar    --repo-type dataset --local-dir .
tar xf TRACE-artifacts-cve.tar      # -> artifacts/doi2/
```

After unpacking, `artifacts/` and `data/cve75/` sit next to `src/`, and every
command below runs as written.

Setup: Python 3.8. Install deps (`pip install -r requirements.txt`, plus
`conda install -c pytorch faiss-gpu=1.7.2`), then:

```bash
source env.sh   # puts src/* on PYTHONPATH
```

## RQ1 — effectiveness (XO / XA / XM, 10K & 100K pools)

One command per cell; `--trust-pool` accepts the downloaded materialized pool.
Metrics land in `<save-db>/recall_eval_metrics.json`.

```bash
python3 src/retrieval/pool_inline_search.py \
  --query-dir artifacts/doi1/query_partial \
  --pool-dir  artifacts/doi1/pool_partial_10K \
  --pool-size 10000 \
  --model artifacts/faiss/inline_slice_pool.index \
  --save-db runs/partial_10K --trust-pool
# XO: query_noinline_cross_opt / pool_cross_opt_{10K,100K}
# XA: query_noinline_cross_arch / pool_cross_arch_{10K,100K}
```

## RQ2 — pairwise verification vs CI-Detector

```bash
python3 src/baselines/pairwise_verification_ci_vs_slice.py \
  --pool-dir artifacts/doi1/pool_partial_10K \
  --query-dir artifacts/doi1/query_partial \
  --ci-pool-dir artifacts/doi1/ci_embeddings/pool \
  --ci-query-dir artifacts/doi1/ci_embeddings/strip \
  --output runs/pairwise_ci_vs_slice.json
```

## RQ3 — ablation (A1 candidate generation / A2 fusion channels / A3 directional similarity)

Same CLI as `pool_inline_search.py`; reuses the RQ1 10K artifacts and prints
all three ablation tables from one run.

```bash
python3 src/ablation/pool_inline_search_ablation.py \
  --query-dir artifacts/doi1/query_partial \
  --pool-dir  artifacts/doi1/pool_partial_10K \
  --pool-size 10000 \
  --model artifacts/faiss/inline_slice_pool.index \
  --save-db runs/ablation_partial_10K --trust-pool
```

## RQ4 — one-day inline CVE detection on a 1M pool

The 1M runtime ships prebuilt (DOI-2). Copy it into your run dir, then search.
Needs a GPU with ≥28 GB free memory for the warm pool.

```bash
cp -al artifacts/doi2/CVE_1M_runtime runs/CVE_1M   # or plain cp
python3 src/retrieval/pool_inline_search.py \
  --query-dir artifacts/doi2/query_CVE \
  --pool-dir  artifacts/doi2/pool_CVE_1M \
  --pool-size 1000000 \
  --model artifacts/faiss/inline_slice_pool.index \
  --save-db runs/CVE_1M --trust-pool

# + symbolic-anchor verifier (uses the prebuilt anchor index in the pool dir):
#   add  --anchor --anchor-query-dir data/cve75/query_strip \
#        --anchor-index artifacts/doi2/pool_CVE_1M/anchor_index_bb_slice_feature.pkl
# + LLM verifier: add  --llm-verify --llm-verify-topk 50  (set $DEEPSEEK_API_KEY)
```

## The cve75 dataset

`data/cve75/` holds the 75 inlined-CVE queries used in RQ4: known-vulnerable
functions inlined into a host in `query_strip/`, with their standalone
ground-truth compilations in `gt_pool/`. `cases.json` gives per-query
provenance (CVE id, library, vulnerable function, host function, source
location, query/GT binary, type). `*_feature.json` holds the decompiled
pseudocode; `*_bb_slice_feature.json` holds the extracted semantic units —
`src/extraction/bb_slice_extractor.py` reproduces the latter from the former.

## What is not released

The encoder checkpoints and training pipeline (group policy), the raw
multi-hundred-GB compile corpus, and IDA-side feature extraction. The paper
documents the training recipe; released embeddings make every experiment
reproducible without the encoder.
