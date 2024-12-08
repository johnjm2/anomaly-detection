import pandas as pd
from pydmd import DMD
import numpy as np

# Load the data from a CSV file
df = pd.read_csv('training_data.csv', parse_dates=['date'])

# Normalize the date to remove the time component
df['date'] = df['date'].dt.normalize()

# Count the number of Connect activities per day
df_connect = df[df['activity'] == 'Connect'].groupby('date').size().reset_index(name='connect_count')

# Prepare the data for DMD: we need a 2D array where each column is a snapshot in time (daily counts)
data_for_dmd = df_connect['connect_count'].values.reshape(1, -1)

# Initialize and fit the DMD model on the entire data
dmd = DMD(svd_rank=-1)
dmd.fit(data_for_dmd)

# Predict future 'Connect' activities for the next 132 days
future_timesteps = 7  # Number of timesteps to predict

# Function to predict future states using DMD
def predict_dmd(timesteps, dmd_model):
    predictions = np.zeros(timesteps, dtype='complex')
    for i in range(timesteps):
        predictions[i] = np.dot(dmd_model.modes, dmd_model.dynamics[:, i])
    return predictions.real  # Extract only the real part of the predictions

# Perform the prediction
predictions = predict_dmd(future_timesteps, dmd)

# Compute the mean and standard deviation of the prediction errors
errors = data_for_dmd.flatten() - dmd.reconstructed_data.flatten().real
error_mean = np.mean(errors)
error_std = np.std(errors)

# Set a threshold for anomaly detection (e.g., mean + 2*std)
threshold = error_mean + 2 * error_std

# Prepare the predictions DataFrame
future_dates = pd.date_range(start=df_connect['date'].iloc[-1] + pd.Timedelta(days=1), periods=future_timesteps)
predictions_df = pd.DataFrame({
    'prediction_date': future_dates,
    'predicted_count': predictions,
    'upper_threshold': predictions + threshold
})

# Ensure that only the real parts are used in the DataFrame
predictions_df['predicted_count'] = predictions_df['predicted_count'].apply(lambda x: np.real(x))
predictions_df['upper_threshold'] = predictions_df['upper_threshold'].apply(lambda x: np.real(x))

# Save the DataFrame to a CSV file
predictions_df.to_csv('dmd_predictions.csv', index=False)

print("Predictions and thresholds saved to 'dmd_predictions.csv'")
