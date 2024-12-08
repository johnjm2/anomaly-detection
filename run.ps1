# PowerShell version of run.sh

# Check if predictions file exists
$predictionsFile = "predictions\dmd_predictions.csv"
if (-not (Test-Path $predictionsFile)) {
    Write-Error "Predictions file not found at $predictionsFile"
    exit 1
}

# Check if streaming data directory exists
$streamingDir = "data\streaming"
if (-not (Test-Path $streamingDir)) {
    Write-Error "Streaming data directory not found at $streamingDir"
    exit 1
}

# Run DMD Analysis
Write-Output "Running DMD Analysis..."
python .\dmd_analysis.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error: DMD Analysis failed."
    exit 1
}
Write-Output "DMD Analysis completed."

# Run PySpark Test
Write-Output "Running PySpark streaming anomaly detection..."
spark-submit .\pyspark_test.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "Error: PySpark streaming failed."
    exit 1
}
Write-Output "PySpark streaming completed successfully."
