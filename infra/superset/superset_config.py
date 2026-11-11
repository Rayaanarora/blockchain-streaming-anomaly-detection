"""
Superset configuration for the local BI layer (docs/BI_SUPERSET.md).

Runs inside the Ubuntu WSL distribution on D: (no Docker). Loaded via
    SUPERSET_CONFIG_PATH=/mnt/d/blockchainbigdata/blockchain-streaming-anomaly-detection/infra/superset/superset_config.py
The secret key is NOT in the repository: it is read from ~/bi/secret_key (created by scripts/bi/setup_superset.sh).
"""
import os

HOME = os.path.expanduser("~")
BI_HOME = os.environ.get("BSAD_BI_HOME", os.path.join(HOME, "bi"))

with open(os.path.join(BI_HOME, "secret_key")) as fh:
    SECRET_KEY = fh.read().strip()

# Superset's own metadata (users, charts, dashboards). SQLite is enough for one local user.
SQLALCHEMY_DATABASE_URI = f"sqlite:///{os.path.join(BI_HOME, 'superset_meta.db')}?check_same_thread=false"

FEATURE_FLAGS = {
    "EMBEDDED_SUPERSET": True,          # dashboards embedded in the FastAPI dashboard via guest tokens
    "DASHBOARD_RBAC": True,
    "ENABLE_TEMPLATE_PROCESSING": True,
}

# Embedded dashboards: guest users get this role (created by setup_superset.sh with read access only)
GUEST_ROLE_NAME = "EmbedViewer"
GUEST_TOKEN_JWT_EXP_SECONDS = 3600

# Local-only deployment: the FastAPI dashboard on 127.0.0.1:8000 frames Superset on 127.0.0.1:8088.
TALISMAN_ENABLED = False
WTF_CSRF_ENABLED = True
WTF_CSRF_EXEMPT_LIST = ["superset.views.core.log", "superset.security.api.guest_token"]
SESSION_COOKIE_SAMESITE = "Lax"
HTTP_HEADERS = {}
ENABLE_CORS = True
CORS_OPTIONS = {"supports_credentials": True, "origins": ["http://127.0.0.1:8000", "http://localhost:8000"]}

# Small machine: keep query workers and result sizes modest
ROW_LIMIT = 50000
SQL_MAX_ROW = 100000
SUPERSET_WEBSERVER_TIMEOUT = 120
CACHE_CONFIG = {"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 60}
DATA_CACHE_CONFIG = {"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 30}
# The default metastore-backed caches fail with the resolved flask-caching version
# (SupersetMetastoreCache got 'ignore_delete_many_errors'); a single local process can use memory caches.
FILTER_STATE_CACHE_CONFIG = {"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 86400}
EXPLORE_FORM_DATA_CACHE_CONFIG = {"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 86400}

# MCP service (Superset's own MCP server), development auth: tool calls run as this local user.
# Local machine only; the server binds to 127.0.0.1.
MCP_AUTH_ENABLED = False
MCP_DEV_USERNAME = "admin"

# ---------------------------------------------------------------- look: match the FastAPI dashboard
# Categorical palette validated for the main dashboard (dataviz validator, dark surface #13161b):
# Elliptic blue, Zenodo orange, aqua, yellow, then illicit red and neutral grey.
EXTRA_CATEGORICAL_COLOR_SCHEMES = [{
    "id": "bsad",
    "description": "BSAD dashboard palette",
    "label": "BSAD",
    "isDefault": True,
    "colors": ["#3987e5", "#d95926", "#199e70", "#c98500", "#e66767", "#8a93a0", "#7b61c9", "#2bb3c0", "#b5487a", "#5f8f3a"],
}]
EXTRA_SEQUENTIAL_COLOR_SCHEMES = [{
    "id": "bsad_blue",
    "description": "BSAD sequential blue",
    "label": "BSAD blue",
    "isDefault": True,
    "colors": ["#1b2430", "#1d3a5c", "#245a94", "#3987e5", "#8cc0ff"],
}]
_FONT = "'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif"
THEME_DEFAULT = {
    "algorithm": "dark",
    "token": {
        "colorPrimary": "#3987e5", "colorBgBase": "#13161b", "colorBgContainer": "#13161b",
        "colorBgLayout": "#0b0d10", "colorBgElevated": "#191d23", "colorTextBase": "#eef0f3",
        "colorBorder": "#2a303a", "colorError": "#e66767", "colorSuccess": "#199e70",
        "colorWarning": "#c98500", "fontFamily": _FONT, "borderRadius": 8,
    },
}
THEME_DARK = THEME_DEFAULT
