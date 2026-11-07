"""
Final Project Part I - Batch pipeline with Spark on AWS EMR
Dataset: Overture Maps buildings (theme=buildings / type=building)

Question we answer: how complete is the 3D info (height / floors) of the world's buildings,
broken down by region, data source and building type?

The main() block only orchestrates: all the real logic lives in SparkUtils (spark_utils.py),
which gets shipped to the cluster with --py-files.
"""
import argparse

from pyspark import StorageLevel
from spark_utils import SparkUtils


def parse_args():
    parser = argparse.ArgumentParser(description="Overture buildings batch pipeline")
    parser.add_argument("--input", required=True, help="S3 folder with the Overture parquet files")
    parser.add_argument("--output", required=True, help="S3 folder where results get written")
    parser.add_argument("--rows", type=int, default=20, help="how many rows to print with show()")
    parser.add_argument("--shuffle-partitions", type=int, default=200, help="spark.sql.shuffle.partitions")
    return parser.parse_args()


def main():
    args = parse_args()
    output = args.output.rstrip("/")

    utils = SparkUtils("overture-buildings-pipeline", shuffle_partitions=args.shuffle_partitions)

    # ---------------- 1. Ingestion ----------------
    raw_df = utils.read_overture_buildings(args.input)
    print("=== Input schema ===")
    raw_df.printSchema()
    # count() on parquet only reads file footers, so it's cheap
    print(f"=== Raw records: {raw_df.count():,} ===")

    # ---------------- 2. Transformations (all lazy) ----------------
    clean_df = SparkUtils.clean_buildings(raw_df)
    derived_df = SparkUtils.derive_columns(clean_df)
    lookup_df = utils.build_source_lookup()
    joined_df = SparkUtils.join_source_lookup(derived_df, lookup_df)
    buildings_df = SparkUtils.filter_relevant(joined_df)

    summary_df = SparkUtils.summarize_by_region_source(buildings_df)
    tiles_df = SparkUtils.top_dense_tiles(buildings_df, n=args.rows)

    # keep the enriched data around so the writes and shows below don't redo the whole job
    buildings_df.persist(StorageLevel.MEMORY_AND_DISK)

    # ---------------- 3. Persistence ----------------
    buildings_path = f"{output}/buildings_enriched"
    summary_path = f"{output}/summary_by_region"

    SparkUtils.write_partitioned_parquet(buildings_df, buildings_path, "source_code")
    SparkUtils.write_partitioned_parquet(summary_df, summary_path, "region")

    # ---------------- 4. Results / validation ----------------
    print("=== Top groups (read back from S3) ===")
    check_df = utils.spark.read.parquet(summary_path)
    check_df.orderBy("n_buildings", ascending=False).show(args.rows, truncate=False)

    print("=== Densest 1x1 degree tiles ===")
    tiles_df.show(args.rows, truncate=False)

    print("=== Sample of enriched buildings (read back from S3) ===")
    utils.spark.read.parquet(buildings_path).show(args.rows, truncate=False)

    buildings_df.unpersist()
    utils.spark.stop()


if __name__ == "__main__":
    main()
