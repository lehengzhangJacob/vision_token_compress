# vision_token_compress

Backup of the PruneVid / VidCom² / AOT reproduction workspace.

## Contents

| Directory | Description |
|-----------|-------------|
| `PruneVid/` | PruneVid core, evaluation scripts, ST-LLM / OV integrations |
| `AOT/` | LLaVA-OneVision (AOT) baseline & PruneVid hooks |
| `VidCom2/` | VidCom² reproduction |
| `VidCom2-qwen/` | VidCom² Qwen variant (worktree copy) |

## Not included (too large for GitHub)

These stay on the local machine only:

- `PruneVid/MODELS/` — model checkpoints (~44 GB)
- `PruneVid/DATAS/` — benchmark videos & annotations (~23 GB)
- `**/logs/`, `**/test_results/` — run logs and raw outputs

Restore data locally with the download scripts under `PruneVid/scripts/` and `PruneVid/download_*.sh`.

## Quick start

```bash
source nk_env.sh          # or PruneVid/prunevid_env.sh
cd PruneVid
bash run_prunevid_extension_queue.sh
```

See `reproduction_report.tex` / `.pdf` for experiment tables.
