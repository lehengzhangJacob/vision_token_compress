#!/usr/bin/env bash
# Marp CLI wrapper (Node installed under ~/.local/node)
export PATH="/home/msj_team/.local/node/bin:$PATH"
exec marp "$@"
