import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, udf, date_format, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType

# Paths
script_dir = os.path.dirname(__file__)
predictions_path = os.path.join(script_dir, 'predictions', 'dmd_predictions.csv')
streaming_data_path = os.path.join(script_dir, 'data', 'streaming')

# Ensure required paths exist
if not os.path.exists(predictions_path):
    raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
if not os.path.exists(streaming_data_path):
    raise FileNotFoundError(f"Streaming data folder not found: {streaming_data_path}")

# Initialize Spark Session
spark = SparkSession.builder.appName("AnomalyDetection").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Log4j Logging
log4jLogger = spark._jvm.org.apache.log4j
logger = log4jLogger.LogManager.getLogger("AnomalyDetection")
logger.info("Application started...")

# Load predictions into a DataFrame
predictions_schema = StructType([
    StructField("prediction_date", StringType(), True),
    StructField("predicted_count", DoubleType(), True),
    StructField("upper_threshold", DoubleType(), True),
])

predictions_df = spark.read.csv(
    predictions_path,
    header=True,
    schema=predictions_schema
).withColumn("prediction_date", to_date(col("prediction_date"), "yyyy-MM-dd"))

logger.info("Predictions DataFrame loaded.")

# Define Anomaly Detection Function
@udf(BooleanType())
def is_anomaly(daily_count, upper_threshold):
    return daily_count > upper_threshold

# Load and Process Streaming Data
streaming_schema = StructType([
    StructField("id", StringType()),
    StructField("date", StringType()),
    StructField("user", StringType()),
    StructField("pc", StringType()),
    StructField("activity", StringType())
])

streaming_df = spark.readStream \
    .schema(streaming_schema) \
    .csv(streaming_data_path)

logger.info("Streaming DataFrame loaded.")

# Normalize and convert the date in streaming data
streaming_df = streaming_df.withColumn(
    "date",
    to_date(to_timestamp(col("date"), "MM/dd/yyyy HH:mm:ss"))
).withColumn(
    "formatted_stream_date",
    date_format(col("date"), "yyyy-MM-dd")
)

# Aggregate the counts of 'Connect' activities per day
daily_counts_df = streaming_df \
    .where(col("activity") == "Connect") \
    .groupBy("formatted_stream_date") \
    .count() \
    .withColumnRenamed("count", "daily_count")

# Join the streaming data with predictions to compare the counts
anomaly_df = daily_counts_df.join(
    predictions_df,
    daily_counts_df.formatted_stream_date == predictions_df["prediction_date"],
    "inner"
).select(
    col("formatted_stream_date"),
    col("daily_count"),
    col("upper_threshold"),
    is_anomaly(col("daily_count"), col("upper_threshold")).alias("is_anomaly")
)

logger.info("Anomaly detection DataFrame created.")

# Write the results to the console
query = anomaly_df.writeStream \
    .outputMode("update") \
    .format("console") \
    .start()

query.awaitTermination()
