#!/bin/bash
cd /home/msj_team/Jacob/nk/VidCom2
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true
echo "=== lmms_eval processes ==="
ps aux | grep -E 'lmms_eval|accelerate' | grep -v grep || echo "(none)"
echo "=== Recent logs ==="
ls -lt logs/repro/ov-7b 2>/dev/null | head -8 || true
ls -lt logs/repro/vid-7b 2>/dev/null | head -8 || true
echo "=== Master log tail ==="
tail -5 logs/run_all.log 2>/dev/null || true
