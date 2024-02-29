import numpy as np
import math


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def root_mean_squared_error(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))


def r_squared(y_true, y_pred):
    mean_true = np.mean(y_true)
    ss_total = np.sum((y_true - mean_true) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_res / ss_total)
    return r2


true_values = np.array([1, 4, 5, 12, 3.2, 3.9])
predicted_values = np.array([1.4, 3.9, 5.5, 11.4, 2.8, 4.4])

mse = mean_squared_error(true_values, predicted_values)
mae = mean_absolute_error(true_values, predicted_values)
rmse = root_mean_squared_error(true_values, predicted_values)

r2 = r_squared(true_values, predicted_values)

print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.2f}")
