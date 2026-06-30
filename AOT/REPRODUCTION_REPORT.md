# AOT (CVPR 2026) Reproduction Report
_Generated 2026-06-25 21:52_  
Model: LLaVA-OneVision-7B / LLaVA-Video-7B - 2x RTX 4090, hf-mirror, training-free AOT.
EgoSchema (full test set) has no public ground truth; validated by prediction-agreement with the authors' committed submission files (100% agreement => identical server score).

### LLaVA-OneVision-7B (Table 1)

| Benchmark | Ratio | Reproduced | Paper | Delta |
|---|---|---|---|---|
| MVBench | 10% | 56.70 | 57.0 | -0.30 |
| MVBench | 15% | 57.33 | 57.8 | -0.47 |
| MVBench | 20% | 57.73 | 58.1 | -0.38 |
| MVBench | 25% | 57.75 | 58.7 | -0.95 |
| EgoSchema | 10% | 98.6% pred-agree (n=5031) | 60.6 | (submission) |
| EgoSchema | 15% | 95.0% pred-agree (n=5031) | 61.3 | (submission) |
| EgoSchema | 20% | 95.8% pred-agree (n=5031) | 61.3 | (submission) |
| EgoSchema | 25% | 94.1% pred-agree (n=5031) | 61.3 | (submission) |
| LongVideoBench | 10% | 52.06 | 54.2 | -2.14 |
| LongVideoBench | 15% | 52.73 | 55.2 | -2.47 |
| LongVideoBench | 20% | 53.18 | 56.2 | -3.02 |
| LongVideoBench | 25% | 52.80 | 56.3 | -3.49 |
| VideoMME | 10% | 53.30 | 56.1 | -2.80 |
| VideoMME | 15% | 53.70 | 56.6 | -2.90 |
| VideoMME | 20% | 58.59 | 57.2 | +1.39 |
| VideoMME | 25% | 54.59 | 57.5 | -2.91 |

### LLaVA-Video-7B (Table 2)

| Benchmark | Ratio | Reproduced | Paper | Delta |
|---|---|---|---|---|
| MVBench | 15% | 57.98 | 57.8 | +0.18 |
| MVBench | 25% | 58.42 | 58.8 | -0.38 |
| EgoSchema | 15% | submission n=5031 | 55.2 | (no GT) |
| EgoSchema | 25% | submission n=5031 | 55.4 | (no GT) |
| LongVideoBench | 15% | 55.42 | 55.0 | +0.42 |
| LongVideoBench | 25% | 56.24 | 56.2 | +0.04 |
| VideoMME | 15% | 62.11 | 62.0 | +0.11 |
| VideoMME | 25% | 62.52 | 62.4 | +0.12 |
