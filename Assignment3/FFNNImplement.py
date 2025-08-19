
# Name: Issa Tijani
# Class: DATA 527
# Term: Spring 2025
# Due Date: 4/3/2025


# XOR Neural Network from Scratch
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid and derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(output):
    return output * (1 - output)

# XOR input and output
data = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
labels = np.array([[0], [1], [1], [0]])

# Set seed for reproducibility
np.random.seed(0)

# Training parameters
lr = 0.5
iterations = 10000

# ----------------------
# Batch Gradient Descent
# ----------------------

# Initialize weights and biases for batch
w1, w2, w3, w4 = np.random.rand(4)
w5, w6 = np.random.rand(2)
b1, b2, b3 = np.random.rand(3)
errors_batch = []

for epoch in range(iterations):
    total_error = 0
    dw1 = dw2 = dw3 = dw4 = dw5 = dw6 = db1 = db2 = db3 = 0

    for x, y in zip(data, labels):
        i1, i2 = x

        # Feedforward
        h1_in = w1*i1 + w3*i2 + b1
        h1_out = sigmoid(h1_in)

        h2_in = w2*i1 + w4*i2 + b2
        h2_out = sigmoid(h2_in)

        o1_in = w5*h1_out + w6*h2_out + b3
        o1_out = sigmoid(o1_in)

        # Error (Loss Metric: MSE)
        error = o1_out - y
        total_error += error**2

        # Backpropagation
        d_o1 = error * sigmoid_derivative(o1_out)
        dw5 += d_o1 * h1_out
        dw6 += d_o1 * h2_out
        db3 += d_o1

        d_h1 = d_o1 * w5 * sigmoid_derivative(h1_out)
        d_h2 = d_o1 * w6 * sigmoid_derivative(h2_out)
        dw1 += d_h1 * i1
        dw3 += d_h1 * i2
        db1 += d_h1

        dw2 += d_h2 * i1
        dw4 += d_h2 * i2
        db2 += d_h2

    # Update weights and biases
    w1 -= lr * dw1
    w2 -= lr * dw2
    w3 -= lr * dw3
    w4 -= lr * dw4
    w5 -= lr * dw5
    w6 -= lr * dw6
    b1 -= lr * db1
    b2 -= lr * db2
    b3 -= lr * db3

    errors_batch.append(total_error.mean())

# -------------------------------
# Stochastic Gradient Descent (SGD)
# -------------------------------

# Re-initialize weights and biases for SGD
w1_s, w2_s, w3_s, w4_s = np.random.rand(4)
w5_s, w6_s = np.random.rand(2)
b1_s, b2_s, b3_s = np.random.rand(3)
errors_sgd = []

for epoch in range(iterations):
    total_error = 0

    for x, y in zip(data, labels):
        i1, i2 = x

        # Feedforward
        h1_in = w1_s*i1 + w3_s*i2 + b1_s
        h1_out = sigmoid(h1_in)

        h2_in = w2_s*i1 + w4_s*i2 + b2_s
        h2_out = sigmoid(h2_in)

        o1_in = w5_s*h1_out + w6_s*h2_out + b3_s
        o1_out = sigmoid(o1_in)

        # Error
        error = o1_out - y
        total_error += error**2

        # Backpropagation
        d_o1 = error * sigmoid_derivative(o1_out)
        dw5_s = d_o1 * h1_out
        dw6_s = d_o1 * h2_out
        db3_s = d_o1

        d_h1 = d_o1 * w5_s * sigmoid_derivative(h1_out)
        d_h2 = d_o1 * w6_s * sigmoid_derivative(h2_out)
        dw1_s = d_h1 * i1
        dw3_s = d_h1 * i2
        db1_s = d_h1

        dw2_s = d_h2 * i1
        dw4_s = d_h2 * i2
        db2_s = d_h2

        # Update weights and biases immediately
        w1_s -= lr * dw1_s
        w2_s -= lr * dw2_s
        w3_s -= lr * dw3_s
        w4_s -= lr * dw4_s
        w5_s -= lr * dw5_s
        w6_s -= lr * dw6_s
        b1_s -= lr * db1_s
        b2_s -= lr * db2_s
        b3_s -= lr * db3_s

    errors_sgd.append(total_error.mean())

# Save model parameters (batch version shown)
with open("NNModelParameters.txt", "w") as f:
    f.write(f"Learning Rate: {lr}\n")
    f.write(f"Iterations: {iterations}\n")
    f.write(f"Final Error (Batch): {errors_batch[-1]}\n")
    f.write(f"Final Error (SGD): {errors_sgd[-1]}\n")
    f.write("Structure: 2 input, 2 hidden, 1 output\n")
    f.write(f"Final Weights (Batch): w1={w1}, w2={w2}, w3={w3}, w4={w4}, w5={w5}, w6={w6}\n")
    f.write(f"Final Weights (SGD): w1={w1_s}, w2={w2_s}, w3={w3_s}, w4={w4_s}, w5={w5_s}, w6={w6_s}\n")

# Plot both errors
plt.plot(errors_batch, label='Batch GD')
plt.plot(errors_sgd, label='Stochastic GD')
plt.title("Cost Error per Iteration")
plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.legend()
plt.savefig("cost_error_vs_iteration.png")
plt.close()

# Predict function using batch weights
def predict(i1, i2):
    # Step 1: Calculate hidden layer node h1
    h1_input = w1 * i1 + w3 * i2 + b1
    h1_output = sigmoid(h1_input)

    # Step 2: Calculate hidden layer node h2
    h2_input = w2 * i1 + w4 * i2 + b2
    h2_output = sigmoid(h2_input)

    # Step 3: Calculate output node o1
    output_input = w5 * h1_output + w6 * h2_output + b3
    output = sigmoid(output_input)

    # Step 4: Return rounded output value
    return round(output[0], 3)

# Predict function using SGD weights
def predict_sgd(i1, i2):
    # Step 1: Calculate hidden layer node h1
    h1_input = w1_s * i1 + w3_s * i2 + b1_s
    h1_output = sigmoid(h1_input)

    # Step 2: Calculate hidden layer node h2
    h2_input = w2_s * i1 + w4_s * i2 + b2_s
    h2_output = sigmoid(h2_input)

    # Step 3: Calculate output node o1
    output_input = w5_s * h1_output + w6_s * h2_output + b3_s
    output = sigmoid(output_input)

    # Step 4: Return rounded output value
    return round(output[0], 3)

# Predictions from both models
print("Predictions using Batch Gradient Descent:")
for x in data:
    print(f"Input: {x} => Predicted: {predict(x[0], x[1])}")

print("\nPredictions using Stochastic Gradient Descent:")
for x in data:
    print(f"Input: {x} => Predicted: {predict_sgd(x[0], x[1])}")
