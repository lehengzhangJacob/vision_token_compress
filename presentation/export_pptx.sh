#!/usr/bin/env bash
# Export Marp markdown -> PPTX (no Python)
# Works on machine with display, or Windows/Mac with Marp CLI + Chrome/Firefox.
set -euo pipefail
export PATH="${HOME}/.local/node/bin:${PATH}"
# proxy (needed on this server for hf etc.; harmless elsewhere)
[ -f "${HOME}/Jacob/0/env.sh" ] && source "${HOME}/Jacob/0/env.sh"

MD="${1:-$(dirname "$0")/video_token_compression_interview.md}"
OUT="${2:-$(dirname "$0")/video_token_compression_interview_marp.pptx}"

marp "$MD" --pptx --no-stdin -o "$OUT"
echo "Wrote $OUT"
