# findspark is only needed on the local Docker setup; on EMR Spark is already on the path,
# so if the import fails we just skip it
try:
    import findspark
    findspark.init()
except Exception:
    pass

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StringType, IntegerType, LongType, ShortType, DoubleType, FloatType, BooleanType, DateType, TimestampType, BinaryType, StructField, ArrayType
from pyspark.sql.functions import count, when, isnull
from pyspark.sql import functions as F


class SparkUtils:

    def __init__(self, app_name, master_url=None, spark_jars=None, spark_packages=None, shuffle_partitions=5):
        spark_builder = (
            SparkSession.builder
            .appName(app_name))

        if master_url is not None:
            spark_builder = spark_builder.master(master_url)

        if spark_jars is not None:
            spark_builder = spark_builder.config("spark.jars", spark_jars)

        if spark_packages is not None:
            spark_builder = spark_builder.config("spark.jars.packages", spark_packages)

        self._spark = spark_builder.getOrCreate()
        # 5 is fine for small local labs, but with 30+ GB we need way more partitions,
        # so the driver can pass a bigger number
        self._spark.conf.set("spark.sql.shuffle.partitions", str(shuffle_partitions))

    @property
    def spark(self):
        return self._spark

    @staticmethod
    def generate_schema(columns_info) -> StructType:
        """
        Generates a list of StructField objects from a list of tuples.

        Args:
            column_info (list of tuples): Each tuple contains (column_name, data_type_string).

        Returns:
            list: A list of StructField objects.
        """
        # Mapping from string type names to PySpark data types
        type_mapping = {
            "string": StringType(),
            "int": IntegerType(),
            "long": LongType(),
            "short": ShortType(),
            "double": DoubleType(),
            "float": FloatType(),
            "boolean": BooleanType(),
            "date": DateType(),
            "timestamp": TimestampType(),
            "binary": BinaryType(),
            "array_int": ArrayType(IntegerType()),
            "array_string": ArrayType(StringType()),
            "struct": StructType()
        }

        struct_fields = []
        for column_info in columns_info:
            col_name  = column_info[0]
            col_type  = column_info[1]

            if col_type == "struct":
                # Expect a third element with the sub-field definitions
                if len(column_info) < 3 or not isinstance(column_info[2], list):
                    raise ValueError(
                        f"Column '{col_name}' is declared as 'struct' but "
                        f"no sub-fields list was provided as the third element."
                    )
                # Recurse to build the nested StructType
                nested_schema = SparkUtils.generate_schema(column_info[2])
                struct_field  = StructField(col_name, nested_schema, True)

            elif col_type not in type_mapping:
                raise ValueError(f"Unsupported data type: '{col_type}' for column '{col_name}'")

            else:
                # Create a StructField for the column
                struct_field = StructField(column_info[0], type_mapping[column_info[1]], True)

            struct_fields.append(struct_field)

        return StructType(struct_fields)

    @staticmethod
    def count_nulls(df):
        return df.select([count(when(isnull(c), c)).alias(c) for c in df.columns])

    # ------------------------------------------------------------------
    # Final project: Overture Maps buildings pipeline
    # Everything below is lazy (transformations only), except write_partitioned_parquet
    # ------------------------------------------------------------------

    def read_overture_buildings(self, path) -> DataFrame:
        """
        Reads the Overture buildings GeoParquet files and keeps only the columns we use.
        We skip the WKB geometry on purpose: it's the heaviest column and the bbox
        already gives us the location, so Parquet's column pruning saves a ton of I/O.
        """
        df = self._spark.read.parquet(path)
        return df.select(
            "id",
            "bbox",
            "version",
            "sources",
            "subtype",
            "class",
            "names",
            "height",
            "num_floors",
            "num_floors_underground",
            "is_underground",
            "has_parts",
            "roof_shape",
        )

    @staticmethod
    def clean_buildings(df) -> DataFrame:
        """
        Data cleaning:
          - drop rows with no id or no bbox (we can't place them on the map)
          - drop rows with coordinates outside valid lat/lon ranges (malformed)
          - drop duplicated ids
          - turn impossible heights/floors into nulls instead of dropping the whole row
          - fill missing subtype with 'unknown'
        """
        center_lat = (F.col("bbox.ymin") + F.col("bbox.ymax")) / 2
        center_lon = (F.col("bbox.xmin") + F.col("bbox.xmax")) / 2

        return (
            df
            .filter(F.col("id").isNotNull() & F.col("bbox").isNotNull())
            .filter(center_lat.between(-90, 90) & center_lon.between(-180, 180))
            .dropDuplicates(["id"])
            # tallest building on earth is ~830 m, so anything above 1000 m is garbage
            .withColumn("height",
                        F.when((F.col("height") > 0) & (F.col("height") <= 1000), F.col("height")))
            .withColumn("num_floors",
                        F.when((F.col("num_floors") > 0) & (F.col("num_floors") <= 200), F.col("num_floors")))
            .withColumn("subtype", F.coalesce(F.col("subtype"), F.lit("unknown")))
        )

    @staticmethod
    def derive_columns(df) -> DataFrame:
        """
        Column derivation:
          - center_lat / center_lon from the bbox
          - footprint_m2: approx bbox area in square meters (degrees -> meters)
          - primary_source: dataset of the first entry in the sources array
          - estimated_height: real height, or num_floors * 3 m when height is missing
          - height_bucket: low / mid / high / skyscraper / unknown
          - has_height_info: 1 if we know height or floors, 0 otherwise
          - region: rough continent from lat/lon boxes (good enough for summaries)
          - tile_1deg: 1x1 degree grid cell, used to find dense areas
          - has_name: 1 if the building has a primary name
        """
        lat = F.col("center_lat")
        lon = F.col("center_lon")

        # meters per degree, longitude shrinks with cos(lat)
        width_m = (F.col("bbox.xmax") - F.col("bbox.xmin")) * F.lit(111320.0) * F.cos(F.radians(lat))
        depth_m = (F.col("bbox.ymax") - F.col("bbox.ymin")) * F.lit(110574.0)

        estimated_height = F.coalesce(F.col("height"), F.col("num_floors") * F.lit(3.0))

        region = (
            F.when(lat < -60, "antarctica")
            .when(lon.between(-170, -30) & (lat < 32.5) & ~((lat > 24.5) & (lon > -98)), "latin_america")
            .when(lon.between(-170, -50), "north_america")
            .when(lon.between(34, 63) & lat.between(12, 42), "middle_east")
            .when(lon.between(-30, 60) & (lat >= 35), "europe")
            .when(lon.between(-20, 52) & (lat < 35), "africa")
            .when((lon >= 110) & (lat < -10), "oceania")
            .when(lon.between(52, 180), "asia")
            .otherwise("other")
        )

        return (
            df
            .withColumn("center_lat", (F.col("bbox.ymin") + F.col("bbox.ymax")) / 2)
            .withColumn("center_lon", (F.col("bbox.xmin") + F.col("bbox.xmax")) / 2)
            .withColumn("footprint_m2", F.round(width_m * depth_m, 2))
            .withColumn("primary_source", F.coalesce(F.col("sources")[0]["dataset"], F.lit("unknown")))
            .withColumn("estimated_height", F.round(estimated_height, 2))
            .withColumn("height_bucket",
                        F.when(F.col("estimated_height").isNull(), "unknown")
                        .when(F.col("estimated_height") < 10, "low")
                        .when(F.col("estimated_height") < 30, "mid")
                        .when(F.col("estimated_height") < 100, "high")
                        .otherwise("skyscraper"))
            .withColumn("has_height_info", F.when(F.col("estimated_height").isNotNull(), 1).otherwise(0))
            .withColumn("region", region)
            .withColumn("tile_1deg",
                        F.concat_ws("_",
                                    F.floor(F.col("center_lat")).cast("int"),
                                    F.floor(F.col("center_lon")).cast("int")))
            .withColumn("has_name", F.when(F.col("names.primary").isNotNull(), 1).otherwise(0))
        )

    def build_source_lookup(self) -> DataFrame:
        """
        Small reference table that classifies each Overture source dataset.
        source_code is short and has no spaces, so it's a clean partition value.
        """
        rows = [
            ("OpenStreetMap",                          "osm",       "community"),
            ("Esri Community Maps",                    "esri",      "community"),
            ("Microsoft ML Buildings",                 "microsoft", "ml_derived"),
            ("Google Open Buildings",                  "google",    "ml_derived"),
            ("Zenodo (doi:10.5281/zenodo.8174931)",    "zenodo",    "ml_derived"),
            ("Instituto Geográfico Nacional (España)", "ign_spain", "authoritative"),
            ("City of Vancouver",                      "vancouver", "authoritative"),
        ]
        return self._spark.createDataFrame(rows, ["primary_source", "source_code", "source_type"])

    @staticmethod
    def join_source_lookup(df, lookup_df) -> DataFrame:
        """
        Join: adds source_code and source_type to every building.
        Left join + broadcast because the lookup is tiny; any dataset we didn't
        list ends up as 'other' instead of being lost.
        """
        return (
            df.join(F.broadcast(lookup_df), on="primary_source", how="left")
            .withColumn("source_code", F.coalesce(F.col("source_code"), F.lit("other")))
            .withColumn("source_type", F.coalesce(F.col("source_type"), F.lit("other")))
        )

    @staticmethod
    def filter_relevant(df) -> DataFrame:
        """
        Filtering: we only care about above-ground buildings with a realistic footprint
        (bigger than 10 m2 and smaller than 1 km2, everything else is noise or a bad bbox).
        Also drops the nested columns we don't need anymore so the output is flat.
        """
        return (
            df
            .filter(~F.coalesce(F.col("is_underground"), F.lit(False)))
            .filter(F.col("footprint_m2").between(10, 1_000_000))
            .select(
                "id", "region", "tile_1deg", "center_lat", "center_lon",
                "primary_source", "source_code", "source_type",
                "subtype", "class", "height", "num_floors", "estimated_height",
                "height_bucket", "has_height_info", "footprint_m2", "has_name",
                "has_parts", "roof_shape",
            )
        )

    @staticmethod
    def summarize_by_region_source(df) -> DataFrame:
        """
        Aggregation: data-quality summary per region, source and building subtype.
        The key metric is pct_with_height: how much of the data is usable for 3D.
        Sorted so the biggest groups come first.
        """
        return (
            df.groupBy("region", "source_code", "source_type", "subtype")
            .agg(
                F.count("*").alias("n_buildings"),
                F.sum("has_height_info").alias("n_with_height"),
                F.round(F.avg("estimated_height"), 2).alias("avg_height_m"),
                F.max("estimated_height").alias("max_height_m"),
                F.round(F.avg("num_floors"), 2).alias("avg_floors"),
                F.round(F.avg("footprint_m2"), 2).alias("avg_footprint_m2"),
                F.round(F.sum("footprint_m2") / 1_000_000, 3).alias("total_footprint_km2"),
            )
            .withColumn("pct_with_height",
                        F.round(F.col("n_with_height") / F.col("n_buildings") * 100, 2))
            .orderBy(F.desc("n_buildings"))
        )

    @staticmethod
    def top_dense_tiles(df, n=20) -> DataFrame:
        """
        Aggregation + sorting: the 1x1 degree tiles with the most buildings.
        Only tiles with at least 1000 buildings count, so tiny tiles don't show up.
        """
        return (
            df.groupBy("tile_1deg", "region")
            .agg(
                F.count("*").alias("n_buildings"),
                F.round(F.avg("estimated_height"), 2).alias("avg_height_m"),
                F.round(F.avg("has_height_info") * 100, 2).alias("pct_with_height"),
            )
            .filter(F.col("n_buildings") >= 1000)
            .orderBy(F.desc("n_buildings"))
            .limit(n)
        )

    @staticmethod
    def write_partitioned_parquet(df, path, partition_col):
        """
        Persistence: writes the DataFrame as Parquet partitioned by one column.
        This is the only method that actually triggers a write.
        """
        (
            df.write
            .mode("overwrite")
            .partitionBy(partition_col)
            .parquet(path)
        )
