#!/usr/bin/env bash
# Upload existing EgoSchema all_results.json to validation server.
set -euo pipefail
cd "$(dirname "$0")"
source prunevid_env.sh
python3 evaluate_egoschema_result.py --result_dir test_results/pllava-7b-prunevid-egoschema/egoschema
