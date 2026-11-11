#!/usr/bin/env bash
# One-time Superset setup inside the Ubuntu WSL distribution (which lives on D:\WSL\Ubuntu).
# Idempotent: safe to rerun. Never prints passwords; they are stored in ~/bi/*.secret (mode 600).
#
#   wsl -d Ubuntu -- bash /mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection/scripts/bi/setup_superset.sh
set -euo pipefail
REPO=/mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection
BI=$HOME/bi
export SUPERSET_CONFIG_PATH=$REPO/infra/superset/superset_config.py
export FLASK_APP=superset
. "$BI/superset-venv/bin/activate"
umask 077

[ -s "$BI/secret_key" ] || python -c "import secrets; print(secrets.token_urlsafe(48))" > "$BI/secret_key"
[ -s "$BI/admin.secret" ] || python -c "import secrets; print(secrets.token_urlsafe(18))" > "$BI/admin.secret"
[ -s "$BI/embed.secret" ] || python -c "import secrets; print(secrets.token_urlsafe(18))" > "$BI/embed.secret"

superset db upgrade >/dev/null
superset fab create-admin --username admin --firstname BSAD --lastname Admin --email admin@localhost \
    --password "$(cat "$BI/admin.secret")" >/dev/null 2>&1 || true
superset init >/dev/null

# DuckDB serving database (views over the Parquet files on D:)
python "$REPO/scripts/bi/build_duckdb.py" --repo "$REPO" --db "$BI/bsad.duckdb"

# Register the DuckDB connection, the read-only embed role and the service user for guest tokens
python - <<'PY'
import os
from superset.app import create_app
app = create_app()
with app.app_context():
    from superset import db, security_manager as sm
    from superset.models.core import Database
    bi = os.path.expanduser("~/bi")
    uri = f"duckdb:///{bi}/bsad.duckdb?access_mode=read_only"
    d = db.session.query(Database).filter_by(database_name="BSAD DuckDB").one_or_none()
    if d is None:
        d = Database(database_name="BSAD DuckDB", sqlalchemy_uri=uri, expose_in_sqllab=True, allow_run_async=False)
        db.session.add(d)
    else:
        d.sqlalchemy_uri = uri
    db.session.commit()

    role = sm.find_role("EmbedViewer") or sm.add_role("EmbedViewer")
    gamma = sm.find_role("Gamma")
    for p in gamma.permissions:
        if p not in role.permissions:
            role.permissions.append(p)
    pv = sm.add_permission_view_menu("all_datasource_access", "all_datasource_access")
    if pv not in role.permissions:
        role.permissions.append(pv)
    svc_role = sm.find_role("Admin")
    if not sm.find_user("embed_service"):
        sm.add_user("embed_service", "Embed", "Service", "embed_service@localhost", [svc_role],
                    password=open(os.path.join(bi, "embed.secret")).read().strip())
    db.session.commit()
    print("superset setup ok: database 'BSAD DuckDB', role EmbedViewer, user embed_service")
PY
