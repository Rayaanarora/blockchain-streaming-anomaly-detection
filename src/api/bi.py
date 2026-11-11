"""
Superset embedding support for the dashboard (docs/BI_SUPERSET.md).

The browser never sees Superset credentials. It asks this API for a short-lived *guest token*; the API
logs in to Superset as the local service user, requests a guest token for one embedded dashboard and
returns only that token. Streaming datasets get a row-level-security clause for the run selected in the
dashboard, so the embedded charts show the same run as the rest of the console.

Configuration:
  data/bi/embed.json                 written by scripts/bi/build_dashboards.py (dashboard -> embedded UUID)
  BSAD_SUPERSET_URL                  default http://127.0.0.1:8088
  BSAD_SUPERSET_USER / _PASSWORD     default: user embed_service, password read once from the Superset
                                     host (~/bi/embed.secret inside WSL); never sent to the browser
"""
import json
import os
import re
import subprocess
import threading
import time
from typing import Optional

import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EMBED_FILE = os.path.join(ROOT, "data", "bi", "embed.json")
RUN_RE = re.compile(r"^run_\d{8}_\d{6}$")


class SupersetEmbedder:
    def __init__(self):
        self.url = os.environ.get("BSAD_SUPERSET_URL", "http://127.0.0.1:8088").rstrip("/")
        self.user = os.environ.get("BSAD_SUPERSET_USER", "embed_service")
        self._password = os.environ.get("BSAD_SUPERSET_PASSWORD")
        self._session: Optional[requests.Session] = None
        self._session_at = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ config
    def config(self) -> dict:
        if not os.path.exists(EMBED_FILE):
            return {"available": False, "reason": "dashboards not built (run scripts/bi/build_dashboards.py)"}
        cfg = json.load(open(EMBED_FILE))
        try:
            ok = requests.get(f"{self.url}/health", timeout=2).status_code == 200
        except requests.RequestException:
            ok = False
        dash = {k: {"title": v["title"], "description": v["description"], "embed_uuid": v["embed_uuid"],
                    "charts": v["charts"]} for k, v in cfg["dashboards"].items()}
        return {"available": ok, "superset_url": self.url, "dashboards": dash,
                "reason": None if ok else f"Superset is not reachable at {self.url} (start scripts\\bi\\start_bi.ps1)"}

    # ------------------------------------------------------------------ auth
    def _read_password(self) -> str:
        if self._password:
            return self._password
        out = subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "bash", "-c", "cat ~/bi/embed.secret"],
                             capture_output=True, text=True, timeout=30)
        if out.returncode != 0 or not out.stdout.strip():
            raise RuntimeError("cannot read the Superset service password; set BSAD_SUPERSET_PASSWORD")
        self._password = out.stdout.strip()
        return self._password

    def _client(self) -> requests.Session:
        # access tokens last 15 min by default; renew well before that
        if self._session is not None and time.time() - self._session_at < 600:
            return self._session
        s = requests.Session()
        r = s.post(f"{self.url}/api/v1/security/login", timeout=15,
                   json={"username": self.user, "password": self._read_password(), "provider": "db", "refresh": True})
        r.raise_for_status()
        s.headers["Authorization"] = f"Bearer {r.json()['access_token']}"
        csrf = s.get(f"{self.url}/api/v1/security/csrf_token/", timeout=15).json()["result"]
        s.headers.update({"X-CSRFToken": csrf, "Referer": self.url})
        self._session, self._session_at = s, time.time()
        return s

    # ------------------------------------------------------------------ guest token
    def guest_token(self, dashboard: str, run: Optional[str]) -> dict:
        cfg = json.load(open(EMBED_FILE))
        if dashboard not in cfg["dashboards"]:
            raise KeyError(dashboard)
        rls = []
        if run:
            if not RUN_RE.match(run):
                raise ValueError("run must look like run_YYYYMMDD_HHMMSS")
            rls = [{"dataset": ds, "clause": f"run_id = '{run}'"} for ds in cfg["stream_dataset_ids"]]
        body = {"user": {"username": "bsad_viewer", "first_name": "BSAD", "last_name": "Viewer"},
                "resources": [{"type": "dashboard", "id": cfg["dashboards"][dashboard]["embed_uuid"]}], "rls": rls}
        with self._lock:
            r = self._client().post(f"{self.url}/api/v1/security/guest_token/", json=body, timeout=15)
            if r.status_code in (401, 403):  # expired session: log in again once
                self._session = None
                r = self._client().post(f"{self.url}/api/v1/security/guest_token/", json=body, timeout=15)
        r.raise_for_status()
        return {"token": r.json()["token"], "run": run}
