"""
Check that every embedded Superset chart renders inside the FastAPI console (http://127.0.0.1:8000).

For each view it waits until all charts in the embedded dashboard have finished loading, then reports
any chart showing an error and saves a full-page screenshot.

    D:\\blockchainbigdata\\tools\\pwvenv\\Scripts\\python.exe scripts\\bi\\verify_embeds.py --out <dir> [--run run_YYYYMMDD_HHMMSS]
"""
import argparse
import json
import os
import sys
import time

from playwright.sync_api import sync_playwright

VIEWS = ["overview", "detection", "models", "data", "batch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report, bad = {}, 0
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        for v in VIEWS:
            page = b.new_page(viewport={"width": 1440, "height": 1000})
            q = f"?run={a.run}" if a.run else ""
            page.goto(f"{a.base}/{q}#/{v}", wait_until="domcontentloaded", timeout=90000)
            frame = None
            for _ in range(60):
                frames = [f for f in page.frames if "/embedded/" in f.url]
                if frames:
                    frame = frames[0]
                    break
                page.wait_for_timeout(1000)
            if frame is None:
                report[v] = {"error": "no embedded dashboard iframe"}
                bad += 1
                page.close()
                continue
            # wait for charts to appear and all loading spinners to clear
            t0 = time.time()
            while time.time() - t0 < 120:
                n = frame.locator(".chart-slice, [data-test='chart-grid-component']").count()
                loading = frame.locator(".loading, [data-test='loading-indicator']").count()
                if n and not loading:
                    break
                page.wait_for_timeout(2000)
            page.wait_for_timeout(4000)
            charts = frame.locator("[data-test='chart-grid-component']")
            rows = []
            for i in range(charts.count()):
                c = charts.nth(i)
                title = (c.locator("[data-test='editable-title'], .header-title, [data-test='slice-header']").first.inner_text(timeout=2000)
                         if c.locator("[data-test='editable-title'], .header-title, [data-test='slice-header']").count() else f"chart {i}")
                txt = c.inner_text(timeout=5000)
                err = None
                for marker in ("Data error", "Unexpected error", "No results were returned", "Guest user cannot"):
                    if marker in txt:
                        err = marker + ": " + txt.split(marker, 1)[1].strip().split("\n")[0][:160]
                rows.append({"chart": title.strip().split("\n")[0], "error": err})
            errs = [r for r in rows if r["error"]]
            bad += len(errs)
            report[v] = {"charts": len(rows), "errors": errs}
            # let the console size the iframe to the dashboard, then capture the whole page
            page.wait_for_timeout(3000)
            page.screenshot(path=os.path.join(a.out, f"view_{v}.png"), full_page=True)
            print(f"{v:<10} {len(rows):>3} charts, {len(errs)} with errors", flush=True)
            for e in errs:
                print(f"    ! {e['chart']}: {e['error']}", flush=True)
            page.close()
        b.close()
    json.dump(report, open(os.path.join(a.out, "verify_report.json"), "w"), indent=2)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
