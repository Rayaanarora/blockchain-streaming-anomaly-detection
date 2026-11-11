#!/usr/bin/env bash
# Start Superset (inside Ubuntu WSL on D:) on http://127.0.0.1:8088. WSL forwards the port to Windows.
#   wsl -d Ubuntu -- bash /mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection/scripts/bi/start_superset.sh
set -euo pipefail
if [ "${1:-}" = "--log" ]; then mkdir -p /mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection/data/bi/logs; exec > /mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection/data/bi/logs/superset.log 2>&1; fi
REPO=/mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection
export SUPERSET_CONFIG_PATH=$REPO/infra/superset/superset_config.py FLASK_APP=superset
. "$HOME/bi/superset-venv/bin/activate"
# refresh the DuckDB views (new gold tables / new streaming runs); safe while Superset is stopped
python "$REPO/scripts/bi/build_duckdb.py" --repo "$REPO" --db "$HOME/bi/bsad.duckdb" >/dev/null
exec gunicorn --bind 0.0.0.0:8088 --workers 1 --threads 8 --timeout 120 --worker-class gthread \
    --access-logfile - "superset.app:create_app()"
