#!/bin/bash
# Concatenate ModelScope LongVideoBench split tar parts and extract videos to the
# layout lmms-eval expects: $HF_HOME/longvideobench/videos/
source /home/msj_team/Jacob/nk/AOT/aot_env.sh
SRC=/home/msj_team/.cache/huggingface/longvideobench_ms_src
DEST=/home/msj_team/.cache/huggingface/longvideobench
mkdir -p "$DEST"
cd "$SRC" || exit 1
echo "=== tar parts present ==="; ls -la videos.tar.part.* | tail -3
# Stream-concatenate split parts directly into tar (no 161G intermediate file)
echo "=== streaming $(ls videos.tar.part.* | wc -l) parts | tar -x -> $DEST ==="
cat videos.tar.part.* | tar -xf - -C "$DEST"
[ -f subtitles.tar ] && echo "=== extracting subtitles ===" && tar -xf subtitles.tar -C "$DEST"
echo "=== result layout ==="; ls "$DEST" | head; echo "videos dir count:"; ls "$DEST/videos" 2>/dev/null | wc -l
