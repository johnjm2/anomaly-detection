# Anomaly Detection with Dynamic Mode Decomposition and Apache Spark

## Overview

This project demonstrates an anomaly detection pipeline using Dynamic Mode Decomposition (DMD) for predictive modeling and Apache Spark for real-time stream processing. It is designed to replicate real-world scenarios involving real-time data ingestion, anomaly detection, and stream processing.

## Key Features

* Dynamic Mode Decomposition (DMD): Predicts future activities based on historical data.
* Anomaly Detection: Identifies deviations by comparing real-time streaming data against DMD predictions.
* Apache Spark Streaming: Handles real-time data streams and detects anomalies on-the-fly.
* Scalability: A pipeline built for large-scale data engineering workflows.


## Setup and Installation

### Prerequisites:

1. **Python 3.8+**
2. **Apache Spark**
3. **pip** for dependency installation

### Installation Steps:

1. Clone this repository:
 ```bash
    git clone https://github.com/your-username/anomaly-detection.git
 ```
 
2. Install required Python libraries:
 ```bash
    pip install -r requirements.txt
 ```
3. Verify installations:
```bash
    spark-submit --version
```

## Usage

### Step 1: Run DMD Analysis

1. Prepare the training dataset (``training_data.csv``) for DMD analysis.
2. Execute the ``dmd_analysis.py`` script to generate predictions
```bash
   python3 dmd_analysis.py
```
3. Output:
- Predictions saved to ```results/dmd_predictions.csv``` 

### Step 2: Simulate Streaming and Detect Anomalies

1. Add test input files to the streaming_data/ folder (e.g., sample_data1.csv).
2. Run the pyspark_test.py script with Spark:
```bash
   spark-submit pyspark_test.py
```
3. Output:
- Anomalies are displayed directly in the terminal console.

### Step 3: Automate the Workflow
To run the entire workflow (DMD analysis + Spark processing), execute the ``run.sh`` script:

```bash
./run.sh
```

## Technologies Used
- **Python**: Core programming language for DMD analysis and data processing.
- **PyKoopman**: Library for Dynamic Mode Decomposition.
- **Apache Spark**: Real-time stream processing.

## Applications
- Real-time stream processing with Apache Spark.
- Predictive modeling with PyDMD.
- Scalable, real-time anomaly detection systems.

## Notes
- The ``StreamingInput`` folder is for testing input data only. Results are displayed in the terminal during execution.
- Customize the dataset or modify the scripts for your specific use cases.

## License
This project is license under the [MIT License](./LICENSE)