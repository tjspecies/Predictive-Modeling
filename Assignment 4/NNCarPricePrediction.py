"""
NNCarPricePrediction1.py
Name: Issa Tijani
Course: DATA 527
Term: Spring 2025
Due Date: April 17, 2025
Assignment: HW4 - Car Price Prediction using Neural Network
"""

import numpy as np
import matplotlib.pyplot as plt
import csv

# ------------------------------------
# Load and clean raw CSV manually
# ------------------------------------
def convert_price(value):
    value = value.strip()
    if "Lakh" in value:
        return float(value.replace(" Lakh", "")) * 1e5
    elif "Crore" in value:
        return float(value.replace(" Crore", "")) * 1e7
    return np.nan

def map_ownership(text):
    mapping = {
        "1st Owner": 1,
        "2nd Owner": 2,
        "3rd Owner": 3,
        "4th Owner": 4,
        "5th Owner": 5,
        "Test Drive Car": 0
    }
    return mapping.get(text.strip(), np.nan)

def load_and_clean(filename):
    rows = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                price = convert_price(row["Car Prices (In rupee)"])
                kms = int(row["kms Driven"].replace(",", "").replace(" kms", ""))
                engine = int(row["Engine"].replace(" cc", ""))
                seats = int(row["Seats"].replace(" Seats", ""))
                ownership = map_ownership(row["Ownership"])
                year = int(row["Manufacture"])

                # Manual one-hot for Fuel Type and Transmission
                fuel = row["Fuel Type"]
                fuel_onehot = [int(fuel == ft) for ft in ["Cng", "Diesel", "Electric", "Lpg", "Petrol"]]

                trans = row["Transmission"]
                trans_onehot = [int(trans == t) for t in ["Automatic", "Manual"]]

                if np.nan in [price, ownership]:
                    continue

                row_vals = [kms, ownership, year, engine, seats] + fuel_onehot + trans_onehot + [price / 83]
                rows.append(row_vals)
            except:
                continue
    return np.array(rows)

data = load_and_clean("Car Price.csv")
np.random.seed(42)
np.random.shuffle(data)

X = data[:, :-1]
y = data[:, -1:]

# ------------------------------------
# Normalize manually
# ------------------------------------
X_min, X_max = X.min(axis=0), X.max(axis=0)
y_min, y_max = y.min(), y.max()

X_norm = (X - X_min) / (X_max - X_min)
y_norm = (y - y_min) / (y_max - y_min)

# Train-test split
split = int(0.8 * len(X_norm))
X_train, X_test = X_norm[:split], X_norm[split:]
y_train, y_test = y_norm[:split], y_norm[split:]

# ------------------------------------
# Neural network from scratch
# ------------------------------------
input_size = X_train.shape[1]
hidden_size = 8
output_size = 1
learning_rate = 0.00001
iterations = 10000

np.random.seed(42)
w1 = np.random.uniform(-1, 1, (input_size, hidden_size))
w2 = np.random.uniform(-1, 1, (hidden_size, output_size))
b1 = np.random.uniform(-1, 1, (1, hidden_size))
b2 = np.random.uniform(-1, 1, (1, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

errors = []
for i in range(iterations):
    # Forward pass
    z1 = np.dot(X_train, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = sigmoid(z2)

    # Calculate error
    error = y_train - a2
    mse = np.mean(np.square(error))
    errors.append(mse)

    # Backpropagation
    d_output = error * sigmoid_derivative(a2)
    d_hidden = d_output.dot(w2.T) * sigmoid_derivative(a1)

    w2 += a1.T.dot(d_output) * learning_rate
    w1 += X_train.T.dot(d_hidden) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    if i % 100 == 0:
        print(f"Epoch {i}: MSE = {mse:.6f}")

# ------------------------------------
# Save error plot and parameters
# ------------------------------------
plt.plot(errors)
plt.title("Training Error Over Time")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.grid(True)
plt.savefig("TrainingErrorPlot.png")

with open("NNCarPriceParameters.txt", "w") as f:
    f.write(f"Learning Rate: {learning_rate}\\n")
    f.write(f"Iterations: {iterations}\\n")
    f.write(f"Final MSE: {mse:.6f}\\n")
    f.write(f"Structure: {input_size}-{hidden_size}-{output_size}\\n")
    f.write("Weights Input-Hidden:\\n")
    f.write(np.array2string(w1, precision=4) + "\\n")
    f.write("Weights Hidden-Output:\\n")
    f.write(np.array2string(w2, precision=4) + "\\n")

# ------------------------------------
# Prediction Function
# ------------------------------------
def predict_price(new_raw_input):
    x_scaled = (new_raw_input - X_min) / (X_max - X_min)
    a1 = sigmoid(np.dot(x_scaled, w1) + b1)
    a2 = sigmoid(np.dot(a1, w2) + b2)
    return a2 * (y_max - y_min) + y_min

# Example prediction
example_input = X[0].reshape(1, -1)
pred = predict_price(example_input)
print(f"Predicted price (USD): ${pred[0][0]:.2f}")

# ------------------------------------
# Evaluation Metrics
# ------------------------------------
predictions = predict_price(X_test)
predictions_rescaled = predictions
y_test_rescaled = y_test * (y_max - y_min) + y_min

# Calculate MAE
mae = np.mean(np.abs(y_test_rescaled - predictions_rescaled))

# Calculate RMSE
rmse = np.sqrt(np.mean(np.square(y_test_rescaled - predictions_rescaled)))

# Calculate R²
ss_total = np.sum(np.square(y_test_rescaled - np.mean(y_test_rescaled)))
ss_residual = np.sum(np.square(y_test_rescaled - predictions_rescaled))
r2 = 1 - (ss_residual / ss_total)

# Print the calculated metrics
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Square Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Optionally save these metrics to the parameters file
with open("NNCarPriceParameters.txt", "a") as f:
    f.write(f"Mean Absolute Error (MAE): {mae:.2f}\\n")
    f.write(f"Root Mean Square Error (RMSE): {rmse:.2f}\\n")
    f.write(f"R² Score: {r2:.2f}\\n")