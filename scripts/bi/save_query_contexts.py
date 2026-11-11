"""
Let Superset's own frontend write each chart's query_context (needed by embedded/guest viewers).

Charts created through the MCP or REST API have form_data but no stored query_context. Embedded
dashboards run as a guest user, and Superset rejects any guest query whose columns/metrics are not in
the chart's stored definition ("Guest user cannot modify chart payload"). Opening a chart in Explore and
saving it makes the Superset frontend build and store the exact query_context it sends; this script does
that for every chart, with the locally installed Chrome (Playwright, channel="chrome", no browser download).

    D:\\blockchainbigdata\\tools\\pwvenv\\Scripts\\python.exe scripts\\bi\\save_query_contexts.py
"""
import json
import subprocess
import sys
import time

import requests
from playwright.sync_api import sync_playwright

SUPERSET = "http://127.0.0.1:8088"


def admin_password():
    return subprocess.run(["wsl.exe", "-d", "Ubuntu", "--", "cat", "/home/katherine/bi/admin.secret"],
                          capture_output=True, text=True, check=True).stdout.strip()


def rest(pw):
    s = requests.Session()
    tok = s.post(f"{SUPERSET}/api/v1/security/login", json={"username": "admin", "password": pw, "provider": "db"}).json()
    s.headers["Authorization"] = f"Bearer {tok['access_token']}"
    return s


def charts(s):
    out, page = [], 0
    while True:
        res = s.get(f"{SUPERSET}/api/v1/chart/?q=(page:{page},page_size:100,columns:!(id,slice_name,viz_type))").json()["result"]
        out += res
        if len(res) < 100:
            return out
        page += 1


def has_qc(s, cid):
    return bool(s.get(f"{SUPERSET}/api/v1/chart/{cid}").json()["result"].get("query_context"))


def main():
    pw = admin_password()
    s = rest(pw)
    todo = [c for c in charts(s) if not has_qc(s, c["id"])]
    print(f"{len(todo)} charts without query_context", flush=True)
    failed = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1000})
        page.goto(f"{SUPERSET}/login/")
        page.fill("input[name=username], #username", "admin")
        page.fill("input[name=password], #password", pw)
        page.keyboard.press("Enter")
        page.wait_for_load_state("networkidle")
        for c in todo:
            cid = c["id"]
            for attempt in range(2):
                try:
                    page.goto(f"{SUPERSET}/explore/?slice_id={cid}", wait_until="domcontentloaded", timeout=90000)
                    page.wait_for_load_state("networkidle", timeout=60000)
                    break
                except Exception:
                    time.sleep(3)
            time.sleep(2)
            if not has_qc(s, cid):  # not auto-generated: save explicitly (overwrite the same chart)
                try:
                    page.get_by_role("button", name="Save").first.click(timeout=15000)
                    page.get_by_role("dialog").get_by_role("button", name="Save").click(timeout=15000)
                    page.wait_for_load_state("networkidle", timeout=60000)
                    time.sleep(1.5)
                except Exception as e:
                    print(f"  ! {c['slice_name']}: {type(e).__name__}", flush=True)
            ok = has_qc(s, cid)
            print(f"  {'ok ' if ok else 'MISSING'} {c['viz_type']:<22} {c['slice_name']}", flush=True)
            if not ok:
                failed.append(c["slice_name"])
        browser.close()
    print(json.dumps({"charts": len(todo), "failed": failed}))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
