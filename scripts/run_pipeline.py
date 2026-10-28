"""
One command for the whole streaming demo: Kafka -> Spark Structured Streaming -> scoring -> dashboard.

    .venv\\Scripts\\python.exe scripts\\run_pipeline.py

What it does, in order:
  1. starts the local Kafka broker (if it is not already running)
  2. creates a fresh topic and run directory (data\\streaming\\run_<timestamp>), so every run starts clean
  3. starts the Spark streaming consumer and waits until it is processing
  4. starts the dashboard (or reuses one already on the port) and opens it in the browser
  5. replays sample Elliptic snapshots and Zenodo transactions into Kafka
  6. prints progress until every transaction is scored, then keeps the dashboard up
Press Ctrl+C to stop. Everything this script started is shut down; a Kafka broker or dashboard
that was already running is left alone.

Logs for each component are written to the run directory (consumer.out, producer_*.out, api.out).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VENV_PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
if os.path.exists(_VENV_PY) and os.path.normcase(os.path.abspath(sys.prefix)) != os.path.normcase(
        os.path.join(ROOT, ".venv")):
    # started with another interpreter (e.g. plain `python`): rerun with the project's virtual environment.
    # Ctrl+C reaches the child too (shared console); wait for its cleanup instead of printing a traceback.
    _child = subprocess.Popen([_VENV_PY, os.path.abspath(__file__), *sys.argv[1:]])
    while True:
        try:
            sys.exit(_child.wait())
        except KeyboardInterrupt:
            continue
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import local_kafka  # noqa: E402

PY = sys.executable


def _say(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _spawn(args, log_path, env=None):
    return subprocess.Popen([PY, *args], cwd=ROOT, env=env, stdout=open(log_path, "ab"), stderr=subprocess.STDOUT)


def _kill(p, grace=20):
    if p is None or p.poll() is not None:
        return
    try:
        p.wait(grace)  # Ctrl+C already reached the child through the shared console
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)


def _tail(path, n=15):
    try:
        return "".join(open(path, encoding="utf-8", errors="replace").readlines()[-n:])
    except OSError:
        return ""


def _read_jsonl(path):
    try:
        return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    except OSError:
        return []


def _api_up(port):
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/api/runs", timeout=2)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--elliptic-steps", type=int, nargs="+", default=[42, 43], help="Elliptic time steps to replay")
    ap.add_argument("--zenodo-limit", type=int, default=3000, help="number of Zenodo transactions to replay")
    ap.add_argument("--zenodo-chunk", default="487000-487999")
    ap.add_argument("--rate", type=float, default=300, help="events per second per producer")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--keep-kafka", action="store_true", help="leave the broker running on exit")
    ap.add_argument("--exit-when-done", action="store_true", help="shut down once every event is scored")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_name, topic = f"run_{stamp}", f"bsad.run.{stamp}"
    run_dir = os.path.join(ROOT, "data", "streaming", run_name)
    os.makedirs(os.path.join(run_dir, "monitoring"), exist_ok=True)
    url = f"http://127.0.0.1:{args.port}/?run={run_name}"
    consumer = api = None
    producers = []
    kafka_started = False
    grace = 20  # after Ctrl+C the children received it too; give Spark time to stop cleanly

    try:
        # 1. Kafka
        if local_kafka.is_up():
            _say("Kafka already running")
        else:
            _say("starting Kafka ...")
            kafka_started = True  # set first: Ctrl+C during start-up must still stop the broker
            local_kafka.start()
        from confluent_kafka.admin import AdminClient, NewTopic
        admin = AdminClient({"bootstrap.servers": local_kafka.BOOTSTRAP, "logger": local_kafka._QUIET})
        admin.create_topics([NewTopic(topic, 3, 1)])[topic].result()
        del admin  # an idle client would log connection errors once the broker stops
        _say(f"topic {topic}, run directory data\\streaming\\{run_name}")

        # 2. consumer
        _say("starting Spark streaming consumer (usually 30-90 s) ...")
        consumer = _spawn(["scripts/streaming_consumer.py", "--topic", topic, "--run-dir", run_dir,
                           "--trigger", "2 seconds", "--max-offsets-per-trigger", "2000"],
                          os.path.join(run_dir, "consumer.out"))
        progress = os.path.join(run_dir, "monitoring", "progress.jsonl")
        deadline = time.time() + 240
        while not any("batch_id" in p for p in _read_jsonl(progress)):
            if consumer.poll() is not None:
                raise SystemExit("consumer exited during startup:\n" + _tail(os.path.join(run_dir, "consumer.out")))
            if time.time() > deadline:
                raise SystemExit("consumer did not start within 240 s; see " + os.path.join(run_dir, "consumer.out"))
            time.sleep(2)
        _say("consumer is processing")

        # 3. dashboard
        if _api_up(args.port):
            _say(f"dashboard already running on port {args.port}")
        else:
            api = subprocess.Popen([PY, "-m", "uvicorn", "src.api.app:app", "--host", "127.0.0.1",
                                    "--port", str(args.port)], cwd=ROOT,
                                   stdout=open(os.path.join(run_dir, "api.out"), "ab"), stderr=subprocess.STDOUT)
            for _ in range(30):
                if _api_up(args.port):
                    break
                time.sleep(1)
        _say(f"dashboard: {url}")
        if not args.no_browser:
            webbrowser.open(url)

        # 4. producers
        _say(f"replaying Elliptic steps {args.elliptic_steps} and {args.zenodo_limit} Zenodo transactions "
             f"at {args.rate:g} events/s each ...")
        common = ["--topic", topic, "--rate", str(args.rate)]
        producers = [
            _spawn(["scripts/replay_producer.py", "elliptic", "--steps", *map(str, args.elliptic_steps), *common,
                    "--manifest", os.path.join(run_dir, "producer_elliptic.json")],
                   os.path.join(run_dir, "producer_elliptic.out")),
            _spawn(["scripts/replay_producer.py", "zenodo", "--block-chunk", args.zenodo_chunk,
                    "--limit", str(args.zenodo_limit), *common,
                    "--manifest", os.path.join(run_dir, "producer_zenodo.json")],
                   os.path.join(run_dir, "producer_zenodo.out")),
        ]

        # 5. progress
        batches = os.path.join(run_dir, "monitoring", "batches.jsonl")
        expected = None
        done = False
        while True:
            time.sleep(5)
            if consumer.poll() is not None:
                raise SystemExit("consumer stopped unexpectedly:\n" + _tail(os.path.join(run_dir, "consumer.out")))
            if expected is None and all(p.poll() is not None for p in producers):
                for name, p in zip(("elliptic", "zenodo"), producers):
                    if p.returncode:
                        raise SystemExit(f"{name} producer failed:\n"
                                         + _tail(os.path.join(run_dir, f"producer_{name}.out")))
                expected = sum(json.load(open(os.path.join(run_dir, f"producer_{n}.json")))["events_sent"]
                               for n in ("elliptic", "zenodo"))
            rows = _read_jsonl(batches)
            written = sum(r.get("results_written", 0) for r in rows)
            flagged = sum(r.get("flagged", 0) for r in rows)
            dlq = sum(r.get("dead_lettered", 0) for r in rows)
            target = f"/{expected}" if expected else ""
            if not done:
                _say(f"scored {written}{target}  flagged {flagged}  dead-lettered {dlq}")
            if expected and written >= expected and not done:
                done = True
                _say(f"all {written} transactions scored ({flagged} flagged). Dashboard: {url}")
                if args.exit_when_done:
                    grace = 0  # nothing is in flight; checkpoints make a hard stop safe (kill/recover is tested)
                    break
                _say("press Ctrl+C to stop everything")
    except KeyboardInterrupt:
        _say("stopping ...")
    finally:
        for p in producers:
            _kill(p, 5)
        _kill(consumer, grace)
        _kill(api, min(grace, 5))
        if kafka_started and not args.keep_kafka:
            try:
                local_kafka.stop()
            except Exception as e:  # noqa: BLE001
                _say(f"could not stop Kafka cleanly: {e}")
        _say(f"results kept in data\\streaming\\{run_name}")


if __name__ == "__main__":
    main()
