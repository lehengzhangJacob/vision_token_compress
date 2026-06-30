#!/bin/bash
# nk 工作区默认代理环境
# 用法: source /home/msj_team/Jacob/nk/nk_env.sh
if [ -f /home/msj_team/Jacob/0/env.sh ]; then
    # shellcheck source=/home/msj_team/Jacob/0/env.sh
    source /home/msj_team/Jacob/0/env.sh
fi
if ! ss -tlnp 2>/dev/null | grep -q ':7890 '; then
    /home/msj_team/Jacob/0/start.sh >/dev/null 2>&1
fi
