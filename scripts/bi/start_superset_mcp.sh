#!/usr/bin/env bash
# Start Superset's official MCP service (Superset 6.1 + fastmcp) on http://127.0.0.1:5008/mcp.
# Development auth (MCP_AUTH_ENABLED = False in superset_config.py): calls run as the local "admin" user.
# Reachable only from this machine (WSL forwards the port to Windows localhost).
#   wsl -d Ubuntu -- bash /mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection/scripts/bi/start_superset_mcp.sh
set -euo pipefail
if [ "${1:-}" = "--log" ]; then mkdir -p /mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection/data/bi/logs; exec > /mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection/data/bi/logs/superset_mcp.log 2>&1; fi
REPO=/mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection
export SUPERSET_CONFIG_PATH=$REPO/infra/superset/superset_config.py FLASK_APP=superset
. "$HOME/bi/superset-venv/bin/activate"
exec superset mcp run --host 0.0.0.0 --port 5008
