"""
Shared SparkSession factory for this Windows workstation.

Findings that drive this module (see docs/DECISIONS.md, D-03/D-04):
  * The shell's JAVA_HOME points to a non-existent ``...\\jdk\\bin``; the working JVM is the
    Java 8 JRE. We resolve JAVA_HOME explicitly instead of relying on the environment.
  * pyspark 3.5.0 bundles Hadoop 3.3.4, whose local filesystem calls winutils.exe for
    permission operations. Without it, ``df.write.parquet`` and Structured Streaming
    checkpoints fail ("HADOOP_HOME and hadoop.home.dir are unset"). We put the pure-Java
    ``hadoop-bare-naked-local-fs`` FileSystem on the driver classpath instead of
    installing an unofficial native binary. Jars come from scripts/fetch_spark_jars.py.
  * All scratch space (spark.local.dir, java.io.tmpdir, TEMP/TMP) is kept on drive D:.
"""
import glob
import os
import sys
from typing import Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SCRATCH = os.path.join(REPO_ROOT, "data", "spark_temp")
JAVA_CANDIDATES = [
    os.environ.get("BSAD_JAVA_HOME", ""),
    os.environ.get("JAVA_HOME", ""),
    r"C:\Program Files\Java\jre1.8.0_451",
    r"C:\Program Files\Java\jdk-17",
]
# Java 8 auto-updates rename the JRE directory (e.g. jre1.8.0_451 -> jre1.8.0_511); accept any installed one.
JAVA_CANDIDATES += sorted(glob.glob(r"C:\Program Files\Java\jre1.8*"), reverse=True)
BARE_FS = "com.globalmentor.apache.hadoop.fs.BareLocalFileSystem"
BARE_AFS = "com.globalmentor.apache.hadoop.fs.BareStreamingLocalFileSystem"


def resolve_java_home() -> str:
    exe = "java.exe" if os.name == "nt" else "java"
    for cand in JAVA_CANDIDATES:
        if cand and os.path.exists(os.path.join(cand, "bin", exe)):
            return cand
    raise RuntimeError("No usable Java runtime found; set BSAD_JAVA_HOME to a JRE/JDK 8, 11 or 17 directory")


def extra_jars(with_kafka: bool) -> list:
    sys.path.insert(0, REPO_ROOT)
    from scripts.fetch_spark_jars import jar_paths
    jars = [p for p in jar_paths() if with_kafka or "hadoop-bare-naked" in p]
    missing = [p for p in jars if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"missing Spark jars {missing}; run: .venv\\Scripts\\python.exe scripts\\fetch_spark_jars.py")
    return jars


def get_spark(app_name: str, master: str = "local[4]", scratch_dir: Optional[str] = None,
              with_kafka: bool = False, shuffle_partitions: int = 8, driver_memory: str = "4g",
              extra_conf: Optional[dict] = None):
    from pyspark.sql import SparkSession

    scratch = os.path.abspath(scratch_dir or DEFAULT_SCRATCH)
    os.makedirs(scratch, exist_ok=True)
    os.environ["JAVA_HOME"] = resolve_java_home()
    os.environ["TEMP"] = os.environ["TMP"] = scratch
    os.environ["SPARK_LOCAL_DIRS"] = scratch
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    # Forward slashes: Spark splits extraJavaOptions with backslash as an escape character, which turned
    # D:\a\b into the drive-relative "D:ab" and scattered JVM temp files into the working directory.
    jvm = (f"-Djava.io.tmpdir={scratch.replace(os.sep, '/')} --add-opens=java.base/java.lang=ALL-UNNAMED "
           "--add-opens=java.base/java.nio=ALL-UNNAMED --add-opens=java.base/sun.nio.ch=ALL-UNNAMED "
           "--add-opens=java.base/java.util=ALL-UNNAMED --add-opens=java.base/java.lang.invoke=ALL-UNNAMED")
    cp = os.pathsep.join(extra_jars(with_kafka))
    builder = (
        SparkSession.builder.appName(app_name).master(master)
        .config("spark.local.dir", scratch)
        .config("spark.sql.warehouse.dir", os.path.join(scratch, "warehouse"))
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.extraClassPath", cp)
        .config("spark.executor.extraClassPath", cp)
        .config("spark.driver.extraJavaOptions", jvm)
        .config("spark.hadoop.fs.file.impl", BARE_FS)
        .config("spark.hadoop.fs.AbstractFileSystem.file.impl", BARE_AFS)
        .config("spark.sql.streaming.checkpointFileManagerClass",
                "org.apache.spark.sql.execution.streaming.FileSystemBasedCheckpointFileManager")
        .config("spark.ui.enabled", "false")
    )
    for key, value in (extra_conf or {}).items():
        builder = builder.config(key, value)
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def to_spark_path(path: str) -> str:
    """Absolute path with forward slashes, accepted by Hadoop on Windows."""
    return os.path.abspath(path).replace("\\", "/")
