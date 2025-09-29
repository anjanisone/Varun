# glue_job_truncate_load_inline.py
import sys
import logging
from pyspark.sql import SparkSession

APP = "glue_job_truncate_load_inline"

logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")
logger = logging.getLogger(APP)


def process_build_spark(catalog_cfg: dict) -> SparkSession:
    """
    Initialize Spark session with Iceberg extensions and catalog configuration.
    """
    b = SparkSession.builder.appName(APP).config(
        "spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions"
    )
    if catalog_cfg["type"] == "glue":
        name = catalog_cfg["glue"]["name"]
        b = (
            b.config(f"spark.sql.catalog.{name}", "org.apache.iceberg.spark.SparkCatalog")
             .config(f"spark.sql.catalog.{name}.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog")
             .config(f"spark.sql.catalog.{name}.warehouse", f"s3://{name}-warehouse/")
             .config(f"spark.sql.defaultCatalog", name)
        )
    else:
        wh = catalog_cfg["hadoop"]["warehouse"]
        b = (
            b.config("spark.sql.catalog.hadoop", "org.apache.iceberg.spark.SparkCatalog")
             .config("spark.sql.catalog.hadoop.type", "hadoop")
             .config("spark.sql.catalog.hadoop.warehouse", wh)
             .config("spark.sql.defaultCatalog", "hadoop")
        )
    return b.getOrCreate()


def process_read_dms_frame(spark: SparkSession, path: str, fmt: str, select_cols=None):
    """
    Read DMS-landed files into a Spark DataFrame.
    """
    df = spark.read.parquet(path) if fmt.lower() == "parquet" else spark.read.format(fmt).load(path)
    if select_cols:
        existing = [c for c in select_cols if c in df.columns]
        df = df.select(*existing)
    return df


def process_create_table_if_needed(spark, target_ident, src_df, enable_catalog=True):
    """
    Create Iceberg table if it does not exist, if catalog registration is enabled.
    """
    if not enable_catalog:
        logger.info(f"Skipping catalog registration for {target_ident}")
        return
    cols = ", ".join([f"`{c}` {src_df.schema[c].dataType.simpleString()}" for c in src_df.columns])
    spark.sql(f"CREATE TABLE IF NOT EXISTS {target_ident} ({cols}) USING ICEBERG")
    logger.info(f"Ensured table in catalog: {target_ident}")


def process_overwrite(df, target_ident):
    """
    Perform truncate & load (overwrite) into Iceberg table.
    """
    df.writeTo(target_ident).overwritePartitions()


def main():
    """
    Process only Non-PK tables using truncate & load.
    """
    cfg = {
        "catalog": {
            "type": "glue",  # "glue" or "hadoop"
            "enable_catalog": True,  # Global toggle for catalog registration
            "glue": {"name": "glue_catalog", "database": "dms_iceberg"},
            "hadoop": {"warehouse": "s3://my-warehouse/iceberg/"}
        },
        "dms": {
            "format": "parquet",
            "op_column": "op",
            "ops": {"insert": "I", "update": "U", "delete": "D"},
            "ts_column": "commit_ts"
        },
        "tables": [
            {
                "name": "customers",
                "source_path": "s3://my-dms-bucket/dms/customers/",
                "target_identifier": "dms_iceberg.customers",
                "has_pk": False,
                "pk": [],
                "select_columns": ["customer_id","name","tier","commit_ts","op"]
            }
        ]
    }

    spark = None
    try:
        spark = process_build_spark(cfg["catalog"])
        for t in cfg["tables"]:
            if t.get("has_pk"):
                continue
            df = process_read_dms_frame(spark, t["source_path"], cfg["dms"]["format"], t.get("select_columns"))
            if "op" in df.columns:
                df = df.drop("op")
            process_create_table_if_needed(spark, t["target_identifier"], df, cfg["catalog"].get("enable_catalog", True))
            process_overwrite(df, t["target_identifier"])
            logger.info(f"{t['name']}: overwrite completed")
        logger.info("Non-PK job completed")
    except Exception as e:
        logger.exception(f"Job failed: {e}")
        raise
    finally:
        if spark:
            spark.stop()


if __name__ == "__main__":
    main()
