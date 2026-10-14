"""
Fetch the extra JVM jars needed for Spark Structured Streaming on this Windows host.

Why: pyspark 3.5.0 bundles Hadoop 3.3.4 whose local filesystem shells out to
winutils.exe for permission calls. Structured Streaming checkpoints therefore
fail on Windows without native Hadoop binaries. Instead of downloading an
unofficial winutils.exe, we use a pure-Java local FileSystem
(com.globalmentor:hadoop-bare-naked-local-fs, Apache-2.0, Maven Central) and
the official Spark Kafka connector.

Every jar is downloaded from Maven Central and verified against the published
SHA-1 checksum. Jars are stored under data/jars/ (on drive D:, git-ignored).
"""
import hashlib
import os
import sys
import urllib.request

MAVEN = "https://repo1.maven.org/maven2"
JARS = [
    ("org.apache.spark", "spark-sql-kafka-0-10_2.12", "3.5.0"),
    ("org.apache.spark", "spark-token-provider-kafka-0-10_2.12", "3.5.0"),
    ("org.apache.kafka", "kafka-clients", "3.4.1"),
    ("org.apache.commons", "commons-pool2", "2.11.1"),
    ("com.globalmentor", "hadoop-bare-naked-local-fs", "0.1.0"),
]
DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "jars")


def jar_paths(jar_dir: str = DEFAULT_DIR):
    return [os.path.abspath(os.path.join(jar_dir, f"{a}-{v}.jar")) for _, a, v in JARS]


def fetch(jar_dir: str = DEFAULT_DIR) -> list:
    os.makedirs(jar_dir, exist_ok=True)
    out = []
    for group, artifact, version in JARS:
        name = f"{artifact}-{version}.jar"
        dest = os.path.abspath(os.path.join(jar_dir, name))
        url = f"{MAVEN}/{group.replace('.', '/')}/{artifact}/{version}/{name}"
        expected = urllib.request.urlopen(url + ".sha1", timeout=60).read().decode().split()[0].strip()
        if not os.path.exists(dest) or hashlib.sha1(open(dest, "rb").read()).hexdigest() != expected:
            print(f"downloading {url}")
            data = urllib.request.urlopen(url, timeout=300).read()
            actual = hashlib.sha1(data).hexdigest()
            if actual != expected:
                raise RuntimeError(f"SHA-1 mismatch for {name}: expected {expected}, got {actual}")
            with open(dest, "wb") as f:
                f.write(data)
        print(f"ok  {name}  sha1={expected}")
        out.append(dest)
    return out


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR)
