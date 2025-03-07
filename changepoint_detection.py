import numpy as np

def detect_anomalies(predictions, actual_values, k=5):
    """Detect top-k anomalies based on MAE."""
    errors = np.abs(predictions - actual_values)
    anomaly_indices = np.argsort(errors)[-k:]
    return anomaly_indices

if __name__ == "__main__":
    predictions = np.load("predictions.npy")
    actual_values = np.load("actual_values.npy")
    anomalies = detect_anomalies(predictions, actual_values, k=5)
    print("Detected anomalies at indices:", anomalies)
