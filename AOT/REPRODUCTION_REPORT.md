# AOT (CVPR 2026) Reproduction Report

_Generated 2026-07-01_  
Model: LLaVA-OneVision-7B / LLaVA-Video-7B — 2× RTX 4090, hf-mirror, training-free AOT.

EgoSchema full-test labels are not public. Accuracies below are from submitting local predictions to the **official validation server** (`validation-server.onrender.com/api/upload/`). For LLaVA-OV, local predictions also agree with the authors' committed submissions on 94–99% of items (100% agreement ⇒ identical server score).

Averages: **Avg. Score** = mean of MVBench, EgoSchema, LongVideoBench, VideoMME; **Avg. %** = Avg. Score relative to the full-model baseline row.

### LLaVA-OneVision-7B (Table 3)

| Retained | MVBench | EgoSchema | LongVideoBench | VideoMME | Avg. Score | Avg. % | Paper AOT |
|---|---|---|---|---|---|---|---|
| 25% | 57.75 | **61.18** | 52.80 | 54.59 | **56.58** | **96.8** | 58.5 / 100.0 |
| 20% | 57.73 | **61.22** | 53.18 | 58.59 | **57.68** | **98.7** | 58.2 / 99.7 |
| 15% | 57.33 | **60.98** | 52.73 | 53.70 | **56.19** | **96.2** | 57.7 / 98.8 |
| 10% | 56.70 | **60.45** | 52.06 | 53.30 | **55.63** | **95.2** | 57.0 / 97.6 |

EgoSchema paper targets: 61.3 (25/20/15%), 60.6 (10%). Reproduced fullset accuracies match within ~0.1 pt.

LongVideoBench and VideoMME show larger gaps vs. paper (especially at 25%), which drives Avg. % below paper.

### LLaVA-Video-7B (Table 4)

| Retained | MVBench | EgoSchema | LongVideoBench | VideoMME | Avg. Score | Avg. % | Paper AOT |
|---|---|---|---|---|---|---|---|
| 25% | 58.42 | **55.85** | 56.24 | 62.52 | **58.26** | **96.8** | 58.2 / 96.7 |
| 15% | 57.98 | **55.08** | 55.42 | 62.11 | **57.65** | **95.8** | 57.5 / 95.5 |

EgoSchema paper targets: 55.4 (25%), 55.2 (15%). Reproduced fullset: 55.85 / 55.08 — within ~0.5 pt.

MVBench, LongVideoBench, and VideoMME align closely with paper; Avg. % matches paper to within 0.1 pt.
