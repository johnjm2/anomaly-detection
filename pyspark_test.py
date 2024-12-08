from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, udf, date_format, to_timestamp
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, BooleanType

# Initialize Spark Session
spark = SparkSession.builder.appName("AnomalyDetection").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load predictions into a DataFrame
predictions_schema = StructType([
    StructField("prediction_date", StringType(), True),
    StructField("predicted_count", DoubleType(), True),
    StructField("upper_threshold", DoubleType(), True),
])

predictions_df = spark.read.csv(
    "dmd_predictions.csv",
    header=True,
    schema=predictions_schema
)

# Ensure the prediction_date is of date type with the format "YYYY-MM-DD"
predictions_df = predictions_df.withColumn("prediction_date", to_date(col("prediction_date"), "yyyy-MM-dd"))


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
    .csv("Streaming data")

# Normalize and convert the date in streaming data to match the format of predictions
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
)

# Select the relevant columns and check for anomalies
anomaly_df = anomaly_df.select(
    col("formatted_stream_date"),
    col("daily_count"),
    col("upper_threshold"),
    is_anomaly(col("daily_count"), col("upper_threshold")).alias("is_anomaly")
)

# Write the results to the console
query = anomaly_df.writeStream \
    .outputMode("update") \
    .format("console") \
    .start()

query.awaitTermination()