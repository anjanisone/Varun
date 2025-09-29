import sys
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, row_number
from pyspark.sql.window import Window

APP = "glue_job_combined_inline"

logging.basicConfig(level=logging.INFO, format="%(asctime)s|%(levelname)s|%(message)s")
logger = logging.getLogger(APP)


def process_build_spark(catalog_cfg: dict) -> SparkSession:
    """
    Initialize a Spark session with Iceberg extensions and catalog configuration.
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
    if fmt.lower() == "parquet":
        df = spark.read.parquet(path)
    else:
        df = spark.read.format(fmt).load(path)
    if select_cols:
        existing = [c for c in select_cols if c in df.columns]
        df = df.select(*existing)
    return df


def process_latest_per_pk(df, pk_cols, ts_col):
    """
    Deduplicate rows to keep the latest record per primary key.
    """
    if ts_col and ts_col in df.columns:
        w = Window.partitionBy(*pk_cols).orderBy(col(ts_col).desc())
        return df.withColumn("__rn", row_number().over(w)).filter(col("__rn") == 1).drop("__rn")
    return df


def process_create_table_if_needed(spark, target_ident, src_df, enable_catalog=True):
    """
    Create Iceberg table if it does not exist, if catalog registration is enabled.
    """
    if not enable_catalog:
        logger.info(f"Skipping catalog registration for {target_ident}")
        return

    cols = ", ".join(
        [f"`{c}` {src_df.schema[c].dataType.simpleString()}" for c in src_df.columns if c != "op"]
    )
    spark.sql(f"CREATE TABLE IF NOT EXISTS {target_ident} ({cols}) USING ICEBERG")
    logger.info(f"Ensured table in catalog: {target_ident}")


def process_merge_cdc(spark, target_ident, staged_df, pk_cols, op_col, ops):
    """
    Merge staged CDC data into Iceberg target table.
    """
    all_cols = [c for c in staged_df.columns if c != op_col]
    set_clause = ", ".join([f"{target_ident}.{c}=s.{c}" for c in all_cols if c not in pk_cols])
    insert_cols = ", ".join(all_cols)
    insert_vals = ", ".join([f"s.{c}" for c in all_cols])
    staged_df.createOrReplaceTempView("staged_view")
    spark.sql(f"""
        MERGE INTO {target_ident} t
        USING staged_view s
        ON {" AND ".join([f"t.{c}=s.{c}" for c in pk_cols])}
        WHEN MATCHED AND s.{op_col} = '{ops["delete"]}' THEN DELETE
        WHEN MATCHED AND s.{op_col} = '{ops["update"]}' THEN UPDATE SET {set_clause}
        WHEN NOT MATCHED AND s.{op_col} IN ('{ops["insert"]}','{ops["update"]}') THEN INSERT ({insert_cols}) VALUES ({insert_vals})
    """)


def process_overwrite_full(df, target_ident):
    """
    Perform truncate & load (overwrite) into Iceberg table.
    """
    (df.drop("op") if "op" in df.columns else df).writeTo(target_ident).overwritePartitions()


def process_table(spark, table_cfg, dms_cfg, enable_catalog=True):
    """
    Process a single table: CDC merge if PK is defined, otherwise truncate & load.
    """
    name = table_cfg["name"]
    logger.info(f"Start {name}")
    df = process_read_dms_frame(spark, table_cfg["source_path"], dms_cfg["format"], table_cfg.get("select_columns"))
    if not df.columns:
        logger.warning(f"{name}: empty schema")
        return

    tgt = table_cfg["target_identifier"]

    if table_cfg["has_pk"]:
        op_col = dms_cfg["op_column"]
        ops = dms_cfg["ops"]
        if op_col not in df.columns:
            raise ValueError(f"{name}: op column '{op_col}' missing")
        df_latest = process_latest_per_pk(df, table_cfg["pk"], dms_cfg.get("ts_column"))
        process_create_table_if_needed(spark, tgt, df_latest, enable_catalog)
        process_merge_cdc(spark, tgt, df_latest, table_cfg["pk"], op_col, ops)
        logger.info(f"{name}: MERGE completed")
    else:
        base_df = df.drop("op") if "op" in df.columns else df
        process_create_table_if_needed(spark, tgt, base_df, enable_catalog)
        process_overwrite_full(base_df, tgt)
        logger.info(f"{name}: overwrite completed")


def main():
    """
    Main entrypoint.
    Loads configuration inline and processes all tables according to PK strategy.
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
                "name": "orders",
                "source_path": "s3://my-dms-bucket/dms/orders/",
                "target_identifier": "dms_iceberg.orders",
                "has_pk": True,
                "pk": ["order_id"],
                "select_columns": ["order_id","customer_id","amount","status","commit_ts","op"]
            },
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
            process_table(spark, t, cfg["dms"], cfg["catalog"].get("enable_catalog", True))
        logger.info("All tables processed successfully")
    except Exception as e:
        logger.exception(f"Job failed: {e}")
        raise
    finally:
        if spark:
            spark.stop()


if __name__ == "__main__":
    main()
