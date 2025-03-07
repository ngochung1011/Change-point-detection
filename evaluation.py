from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(predictions, actual_values):
    """Compute MAE and MSE for model evaluation."""
    mae = mean_absolute_error(actual_values, predictions)
    mse = mean_squared_error(actual_values, predictions)
    return mae, mse

if __name__ == "__main__":
    predictions = np.load("predictions.npy")
    actual_values = np.load("actual_values.npy")
    mae, mse = evaluate_model(predictions, actual_values)
    print(f"MAE: {mae}, MSE: {mse}")