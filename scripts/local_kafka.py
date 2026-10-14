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


def is_up(timeout=3.0) -> bool:
    from confluent_kafka.admin import AdminClient
    try:
        AdminClient({"bootstrap.servers": BOOTSTRAP}).list_topics(timeout=timeout)
        return True
    except Exception:
        return False


def start():
    if is_up():
        print("kafka already running at", BOOTSTRAP)
        return
    cfg = _config()
    env = _env()
    if not os.path.exists(os.path.join(DATA, "meta.properties")):
        os.makedirs(DATA, exist_ok=True)
        subprocess.run(_java("kafka.tools.StorageTool", "format", "-t", CLUSTER_ID, "-c", cfg), env=env, check=True)
    out = open(os.path.join(RUN, "broker.out"), "ab")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)
    p = subprocess.Popen(_java("kafka.Kafka", cfg), env=env, stdout=out, stderr=out, creationflags=flags)
    open(os.path.join(RUN, "broker.pid"), "w").write(str(p.pid))
    for _ in range(60):
        if is_up(2.0):
            print(f"kafka up at {BOOTSTRAP} (pid {p.pid}, data {DATA})")
            return
        time.sleep(2)
    raise RuntimeError(f"kafka did not come up; see {os.path.join(RUN, 'broker.out')}")


def create_topics():
    from confluent_kafka.admin import AdminClient, NewTopic
    admin = AdminClient({"bootstrap.servers": BOOTSTRAP})
    existing = set(admin.list_topics(timeout=10).topics)
    new = [NewTopic(t, num_partitions=n, replication_factor=1) for t, n in TOPICS.items() if t not in existing]
    for t, f in (admin.create_topics(new).items() if new else []):
        f.result()
        print("created topic", t)
    print("topics:", sorted(t for t in admin.list_topics(timeout=10).topics if not t.startswith("__")))


def stop():
    pid_file = os.path.join(RUN, "broker.pid")
    if os.path.exists(pid_file):
        pid = open(pid_file).read().strip()
        # The broker is the java.exe we launched; without /F Windows asks it to close (no console -> forced).
        subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)
        os.remove(pid_file)
    for _ in range(15):
        if not is_up(1.0):
            print("kafka stopped")
            return
        time.sleep(1)
    raise RuntimeError("kafka still reachable after stop")


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
