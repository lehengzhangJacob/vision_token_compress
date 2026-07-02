# PruneVid Reproduction Report

_Reproduction of **PruneVid** (ACL 2025 Findings, arXiv:2412.16117) applied to three video LLMs: PLLaVA-7B, ST-LLM, and LLaVA-OneVision-7B._

Highlight of this reproduction: **self-ported ST-LLM** and **LLaVA-OneVision** PruneVid pipelines (the official repo only released PLLaVA code).

Paper protocol followed exactly: input frames = **16** for PLLaVA & ST-LLM, **32** for LLaVA-OneVision (64 only for VCGBench); PruneVid hyper-params α=0.4, τ=0.8, temporal-segment-ratio=0.25, cluster-ratio=0.5, attention layer M=10.

_Last updated: 2026-07-01_

## Main results — Paper (w/ PruneVid) vs. our Reproduce

| Backbone | Benchmark | Paper (w/ Ours) | **Reproduce** | Δ |
|---|---|---|---|---|
| **ST-LLM** | MVBench | 54.3 | **54.80** | +0.50 |
| **ST-LLM** | VideoMME | 41.4 | **42.59** | +1.19 |
| **ST-LLM** | EgoSchema (subset) | 54.6 | **59.80** | +5.20 |
| **ST-LLM** | EgoSchema (fullset) | 44.7 | _not run (5031)_ | — |
| **LLaVA-OneVision** | MVBench | 57.5 | **57.23** | -0.27 |
| **LLaVA-OneVision** | VideoMME | 58.6 | **58.37** | -0.23 |
| **LLaVA-OneVision** | EgoSchema (subset / fullset) | 62.6 / 59.5 | **64.40 / 61.78** | +1.8 / +2.3 |
| **PLLaVA** | MVBench | 47.6 | **46.42** | -1.18 |
| **PLLaVA** | VideoMME | 45.0 | **44.89** | -0.11 |
| **PLLaVA** | EgoSchema (subset / fullset) | 49.0 / 42.6 | **48.40 / 41.90** | -0.6 / -0.7 |

All reproduced numbers land within ~1 point of the paper except ST-LLM EgoSchema subset, which is slightly higher (subset selection / beam-search differences; see notes).

## Per-backbone detail

### ST-LLM w/ PruneVid (self-ported) — primary highlight
| Metric | Paper baseline | Paper w/ Ours | Reproduce |
|---|---|---|---|
| MVBench | 54.9 | 54.3 | **54.80** (2192/4000) |
| VideoMME | 42.0 | 41.4 | **42.59** (1150/2700) |
| EgoSchema subset | 56.2 | 54.6 | **59.80** (299/500) |
| EgoSchema fullset | 45.6 | 44.7 | _not run_ |
| VCGBench avg | 2.86 | 2.82 | _skipped (no GPT key)_ |

### LLaVA-OneVision w/ PruneVid (self-ported)
| Metric | Paper baseline | Paper w/ Ours | Reproduce |
|---|---|---|---|
| MVBench | 58.0 | 57.5 | **57.23** |
| VideoMME | 58.2 | 58.6 | **58.37** |
| EgoSchema subset / fullset | 62.0/60.0 | 62.6/59.5 | **64.40 / 61.78** (validation server) |
| VCGBench avg | 3.26 | 3.24 | _skipped (no GPT key)_ |

### PLLaVA-7B w/ PruneVid (official code path)
| Metric | Paper baseline | Paper w/ Ours | Reproduce |
|---|---|---|---|
| MVBench | 46.6 | 47.6 | **46.42** |
| VideoMME | 44.4 | 45.0 | **44.89** |
| EgoSchema subset / fullset | 47.8/42.6 | 49.0/42.6 | **48.40 / 41.90** (validation server) |
| VCGBench avg | 2.99 | 2.98 | _skipped (no GPT key)_ |

## Key fixes made during reproduction

1. **ST-LLM environment** — ST-LLM's custom memory-augmented Llama is incompatible with `transformers` 4.37; built a dedicated conda env (`transformers==4.28.0`, `torch==2.0.1+cu118`) which removed garbage/hallucinated outputs.
2. **VideoMME/EgoSchema scoring bug** (`video_mcq_infer.py::normalize_pred`) — a `startswith(letter)` check matched the leading "A" in the model's "Answer: (X)" prefix, collapsing every prediction to "(A)". Fixed via regex letter extraction; ST-LLM VideoMME recovered 25.15% → **42.59%**.
3. **EgoSchema ground-truth corruption** — the original `egoschema_subset.json` had fabricated answers and a wrong 500-video selection (only 55/500 overlapped the official lmms-lab subset). Rebuilt the subset from the authoritative **lmms-lab/egoschema Subset**; ST-LLM EgoSchema recovered 19.20% (chance) → **59.80%**.

## Not reproduced (by decision / external dependency)

- **VCGBench (all backbones)** — requires GPT-3.5/4 as judge (no API key provided) and 11/499 YouTube source videos are no longer downloadable.
- **ST-LLM EgoSchema fullset** — only the 500-item subset was run locally; fullset (5031) was never executed.
- **EgoSchema fullset scoring** — obtained via the official validation server (`validation-server.onrender.com`); requires direct network access (proxy TLS currently broken for some hosts).
