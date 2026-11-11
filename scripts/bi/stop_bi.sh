#!/usr/bin/env bash
# Stop Superset web and MCP service started by start_bi.ps1 / start_superset*.sh
for p in $(pgrep -f "gunicorn --bind 0.0.0.0:8088"); do kill "$p"; done
for p in $(pgrep -f "superset mcp run"); do kill "$p"; done
sleep 2
pgrep -af "gunicorn --bind 0.0.0.0:8088|superset mcp run" >/dev/null && echo "still running" || echo "BI layer stopped"
