"""
============================================================
  DRIFTING ORACLE — Spark Session Factory
============================================================
  Creates a local SparkSession with Delta Lake support for
  VS Code development. On Databricks, the session already
  exists via the 'spark' variable.
============================================================
"""

import os
import sys

# Add project root to path so config is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.config import IS_DATABRICKS, OUTPUT_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, VECTOR_DB_DIR


def get_spark_session(app_name: str = "DriftingOracle"):
    """
    Returns a SparkSession.
    - On Databricks: returns the existing 'spark' session.
    - On Local: creates a new session with Delta Lake support.
    """
    if IS_DATABRICKS:
        # Databricks already provides a SparkSession as 'spark'
        # This import only works inside Databricks runtime
        from pyspark.sql import SparkSession
        return SparkSession.getActiveSession()

    from pyspark.sql import SparkSession

    import ctypes
    from ctypes import wintypes
    _GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    _GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    _GetShortPathNameW.restype = wintypes.DWORD
    def get_short_path(path):
        output_buf_size = 0
        while True:
            output_buf = ctypes.create_unicode_buffer(output_buf_size)
            needed = _GetShortPathNameW(path, output_buf, output_buf_size)
            if output_buf_size >= needed: return output_buf.value
            else: output_buf_size = needed

    # Fix for Windows PySpark runtime dependencies dynamically!
    short_python = get_short_path(sys.executable)
    os.environ["PYSPARK_PYTHON"] = short_python
    os.environ["PYSPARK_DRIVER_PYTHON"] = short_python
    os.environ["PYSPARK_USE_DAEMON"] = "0"
    os.environ["JAVA_HOME"] = os.path.join(PROJECT_ROOT, "java", "jdk-11.0.21+9")
    os.environ["HADOOP_HOME"] = os.path.join(PROJECT_ROOT, "hadoop")
    os.environ["PATH"] = os.path.join(PROJECT_ROOT, "hadoop", "bin") + os.pathsep + os.environ.get("PATH", "")

    # Create output directories
    for d in [OUTPUT_DIR, BRONZE_DIR, SILVER_DIR, GOLD_DIR, VECTOR_DB_DIR]:
        os.makedirs(d, exist_ok=True)

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[1]")
        .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config(
            "spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog",
        )
        # Performance tuning for local
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.sql.adaptive.enabled", "true")
        # Delta Lake defaults
        .config("spark.databricks.delta.schema.autoMerge.enabled", "true")
        .getOrCreate()
    )

    # Reduce log noise
    spark.sparkContext.setLogLevel("WARN")

    print(f"✅ SparkSession created: {app_name}")
    print(f"   Spark version: {spark.version}")
    print(f"   Delta Lake enabled: True")

    return spark


def save_table(df, path_or_table: str, mode: str = "overwrite", fmt: str = "delta"):
    """
    Save a DataFrame to the correct location.
    - Local:      Writes Delta/Parquet to the file path.
    - Databricks: Writes to Unity Catalog table.
    """
    if IS_DATABRICKS:
        df.write.mode(mode).saveAsTable(path_or_table)
        print(f"✅ Saved to Unity Catalog: {path_or_table}")
    else:
        try:
            df.write.format(fmt).mode(mode).save(path_or_table)
            print(f"✅ Saved as {fmt}: {path_or_table}")
        except Exception as e:
            # Fallback to parquet if delta-spark is not available
            print(f"⚠️  Delta write failed ({e}), falling back to parquet...")
            df.write.format("parquet").mode(mode).save(path_or_table)
            print(f"✅ Saved as parquet: {path_or_table}")


def read_table(spark, path_or_table: str, fmt: str = "delta"):
    """
    Read a table from the correct location.
    - Local:      Reads Delta/Parquet from the file path.
    - Databricks: Reads from Unity Catalog table.
    """
    if IS_DATABRICKS:
        return spark.read.table(path_or_table)
    else:
        try:
            return spark.read.format(fmt).load(path_or_table)
        except Exception:
            # Fallback to parquet
            return spark.read.format("parquet").load(path_or_table)


if __name__ == "__main__":
    spark = get_spark_session("DriftingOracle_Test")
    print(f"\nSpark UI: {spark.sparkContext.uiWebUrl}")
    print("Press Ctrl+C to stop...")
    spark.stop()
    print("🛑 Spark stopped.")
