"""
Manage the local single-node Kafka broker used for replay tests.

    .venv\\Scripts\\python.exe scripts\\local_kafka.py start     # formats storage on first run
    .venv\\Scripts\\python.exe scripts\\local_kafka.py status
    .venv\\Scripts\\python.exe scripts\\local_kafka.py topics    # create the replay topics
    .venv\\Scripts\\python.exe scripts\\local_kafka.py stop

Environment (see .env.example):
  BSAD_KAFKA_HOME   Kafka distribution (default D:\\blockchainbigdata\\tools\\kafka_2.13-3.7.2)
  BSAD_KAFKA_DATA   broker data dir   (default <repo>\\data\\kafka_data)
  BSAD_JAVA_HOME    JRE for the broker (default: resolved like Spark, Java 8 JRE)
Everything (binaries, data, logs) lives on drive D:.
"""
import os
import subprocess
import sys
import time
from typing import Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.spark.session import resolve_java_home  # noqa: E402

KAFKA_HOME = os.environ.get("BSAD_KAFKA_HOME", r"D:\blockchainbigdata\tools\kafka_2.13-3.7.2")
DATA = os.path.abspath(os.environ.get("BSAD_KAFKA_DATA", os.path.join(ROOT, "data", "kafka_data")))
RUN = os.path.join(ROOT, "data", "kafka_run")
BOOTSTRAP = os.environ.get("BSAD_KAFKA_BOOTSTRAP", "127.0.0.1:29092")
TOPICS = {"bsad.events.v1": 3, "bsad.events.dlq.v1": 1}
CLUSTER_ID = "bsadLocalReplay00000A"  # fixed 22-char base64 id -> reproducible storage format


def _env():
    env = dict(os.environ)
    env["JAVA_HOME"] = resolve_java_home()
    env["KAFKA_HEAP_OPTS"] = "-Xms512m -Xmx1g"
    env["LOG_DIR"] = os.path.join(RUN, "logs")
    env["TEMP"] = env["TMP"] = os.path.join(RUN, "tmp")
    os.makedirs(env["LOG_DIR"], exist_ok=True)
    os.makedirs(env["TEMP"], exist_ok=True)
    return env


def _java(main_class, *args):
    """Run Kafka's JVM entry point directly. The bundled .bat launchers concatenate every jar
    into CLASSPATH and exceed cmd.exe's line limit on Windows ("The input line is too long")."""
    java = os.path.join(resolve_java_home(), "bin", "java.exe")
    logs = os.path.join(RUN, "logs")
    return [java, "-Xms512m", "-Xmx1g", "-XX:+UseG1GC",
            f"-Dlog4j.configuration=file:{os.path.join(KAFKA_HOME, 'config', 'log4j.properties')}",
            f"-Dkafka.logs.dir={logs}", f"-Djava.io.tmpdir={os.path.join(RUN, 'tmp')}",
            "-cp", os.path.join(KAFKA_HOME, "libs", "*"), main_class, *args]


def _config():
    src = os.path.join(ROOT, "infra", "kafka", "server.properties")
    os.makedirs(RUN, exist_ok=True)
    dst = os.path.join(RUN, "server.properties")
    text = open(src).read().replace("__SET_BY_LOCAL_KAFKA_PY__", DATA.replace("\\", "/"))
    open(dst, "w").write(text)
    return dst


import logging  # noqa: E402

_QUIET = logging.getLogger("bsad.kafka.probe")
_QUIET.addHandler(logging.NullHandler())
_QUIET.propagate = False


def is_up(timeout=3.0) -> bool:
    from confluent_kafka.admin import AdminClient
    try:
        # a "down" broker is an expected answer here: send librdkafka's connect errors to a silent logger
        AdminClient({"bootstrap.servers": BOOTSTRAP, "logger": _QUIET}).list_topics(timeout=timeout)
        return True
    except Exception:
        return False


def _pid() -> Optional[int]:
    path = os.path.join(RUN, "broker.pid")
    try:
        return int(open(path).read().strip())
    except (OSError, ValueError):
        return None


def _alive(pid: int) -> bool:
    import psutil
    try:
        return psutil.Process(pid).is_running() and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _send_ctrl_c(pid: int) -> bool:
    """Deliver CTRL_C_EVENT to the broker's own (hidden) console so the JVM runs its shutdown hooks.

    Done from a throwaway helper process: it detaches from our console, attaches to the broker's,
    ignores Ctrl+C itself and broadcasts the event to that console only.
    """
    helper = ("import ctypes,sys;k=ctypes.windll.kernel32;k.FreeConsole();"
              f"ok=k.AttachConsole({pid});k.SetConsoleCtrlHandler(None,1);"
              "ok=ok and k.GenerateConsoleCtrlEvent(0,0);sys.exit(0 if ok else 1)")
    r = subprocess.run([sys.executable, "-c", helper], capture_output=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    return r.returncode == 0


def start():
    if is_up():
        print("kafka already running at", BOOTSTRAP)
        return
    cfg = _config()
    env = _env()
    if not os.path.exists(os.path.join(DATA, "meta.properties")):
        os.makedirs(DATA, exist_ok=True)
        subprocess.run(_java("kafka.tools.StorageTool", "format", "-t", CLUSTER_ID, "-c", cfg), env=env, check=True)
    log_path = os.path.join(RUN, "broker.out")
    out = open(log_path, "ab")
    clean = os.path.exists(os.path.join(DATA, ".kafka_cleanshutdown"))
    # Own console (hidden window) instead of DETACHED_PROCESS: stop() can then send it a real Ctrl+C,
    # so Kafka shuts down cleanly and the next start skips log recovery.
    si = subprocess.STARTUPINFO()
    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    si.wShowWindow = 0  # SW_HIDE
    # The "ignore Ctrl+C" console flag is inherited by child processes; some launchers (IDE terminals,
    # Git Bash, services) set it. Clear it so the broker can receive the Ctrl+C that stop() sends.
    import ctypes
    ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
    p = subprocess.Popen(_java("kafka.Kafka", cfg), env=env, stdout=out, stderr=out, startupinfo=si,
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
    open(os.path.join(RUN, "broker.pid"), "w").write(str(p.pid))
    note = "" if clean else " (previous shutdown was not clean: Kafka re-checks every topic log, ~1 min)"
    print(f"waiting for Kafka to start{note} ...", flush=True)
    t0 = time.time()
    last = 0
    while time.time() - t0 < 240:
        if p.poll() is not None:
            tail = "".join(open(log_path, encoding="utf-8", errors="replace").readlines()[-15:])
            raise RuntimeError(f"Kafka exited during startup (code {p.returncode}); last log lines:\n{tail}")
        if is_up(2.0):
            print(f"kafka up at {BOOTSTRAP} after {time.time() - t0:.0f}s (pid {p.pid}, data {DATA})", flush=True)
            return
        if time.time() - last >= 10:
            if last:
                print(f"  still starting ({time.time() - t0:.0f}s) ...", flush=True)
            last = time.time()
        time.sleep(1)
    raise RuntimeError(f"kafka did not come up within 240 s; see {log_path}")


def create_topics():
    from confluent_kafka.admin import AdminClient, NewTopic
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})
    existing = set(admin.list_topics(timeout=10).topics)
    new = [NewTopic(t, num_partitions=n, replication_factor=1) for t, n in TOPICS.items() if t not in existing]
    for t, f in (admin.create_topics(new).items() if new else []):
        f.result()
        print("created topic", t)
    print("topics:", sorted(t for t in admin.list_topics(timeout=10).topics if not t.startswith("__")))


def stop(timeout: float = 60.0):
    """Clean shutdown via Ctrl+C to the broker's console; force-kill only as a fallback."""
    pid_file = os.path.join(RUN, "broker.pid")
    pid = _pid()
    if pid and _alive(pid):
        how = "forced"
        if _send_ctrl_c(pid):
            t0 = time.time()
            while time.time() - t0 < timeout and _alive(pid):
                time.sleep(0.5)
            if not _alive(pid):
                how = "clean"
        if how == "forced":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        print(f"kafka stopped ({how} shutdown)")
    if os.path.exists(pid_file):
        os.remove(pid_file)
    for _ in range(15):
        if not is_up(1.0):
            return
        time.sleep(1)
    raise RuntimeError("kafka still reachable after stop (started by something other than this script?)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "start":
        start()
    elif cmd == "stop":
        stop()
    elif cmd == "topics":
        create_topics()
    elif cmd == "status":
        print("UP" if is_up() else "DOWN", BOOTSTRAP)
    else:
        raise SystemExit(__doc__)
