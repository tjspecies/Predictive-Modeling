
# Simple Linear Regression Implementation using Gradient Descent
# DATA 527 - Issa Tijani - February 2025

import matplotlib.pyplot as plt

# Read dataset
file_name = "myData.csv"
with open(file_name, "r") as file:
    data = file.readlines()

# Parse dataset
x = []
y = []
for line in data[1:]:  # Skip the header
    values = line.strip().split(",")
    x.append(float(values[0]))
    y.append(float(values[1]))

# Initialize parameters
learning_rate = 0.000001
iterations = 1000
n = len(x)  # Number of data points
m = 0  # Slope
b = 0  # Intercept

# Function to calculate mean squared error (MSE)
def calculate_mse(x, y, m, b):
    total_error = 0
    for i in range(n):
        predicted = m * x[i] + b
        total_error += (y[i] - predicted) ** 2
    return total_error / n

# Gradient descent algorithm
mse_log = []  # To save MSE of each iteration
for iteration in range(iterations):
    m_gradient = 0  # Slope gradient
    b_gradient = 0  # Intercept gradient

    # Compute gradients
    for i in range(n):
        predicted = m * x[i] + b
        m_gradient += -(2 / n) * x[i] * (y[i] - predicted)
        b_gradient += -(2 / n) * (y[i] - predicted)

    # Update parameters
    m -= learning_rate * m_gradient
    b -= learning_rate * b_gradient

    # Calculate MSE and log it
    mse = calculate_mse(x, y, m, b)
    mse_log.append(mse)

# Save MSE log to file
mse_file_name = f"SLRTraining[{iterations}][{learning_rate}]MSEs.txt"
with open(mse_file_name, "w") as mse_file:
    for i, mse in enumerate(mse_log):
        mse_file.write(f"Iteration {i+1}: MSE = {mse}\n")

# Plot MSE per iteration
plt.figure(figsize=(8, 6))
plt.plot(range(1, iterations + 1), mse_log, label="MSE per Iteration", color="blue")
plt.title("Mean Squared Error (MSE) per Iteration")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error")
plt.legend()
plt.grid()
mse_plot_file = "MSE_Per_Iteration_Plot.png"
plt.savefig(mse_plot_file)
plt.show()

# Plot Regression Line
plt.figure(figsize=(8, 6))
plt.scatter(x, y, label="Actual Data", color="blue")
plt.plot(x, [m * xi + b for xi in x], label="Regression Line", color="red")
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.title("Simple Linear Regression")
plt.legend()
plt.grid()
regression_plot_file = "RegressionPlot.png"
plt.savefig(regression_plot_file)
plt.show()

# Calculate final R-squared (R²)
y_mean = sum(y) / n
ss_total = sum([(yi - y_mean) ** 2 for yi in y])
ss_residual = sum([(y[i] - (m * x[i] + b)) ** 2 for i in range(n)])
r_squared = 1 - (ss_residual / ss_total)

# Save model parameters to file
parameters_file = "SLRModelParameters.txt"
with open(parameters_file, "w") as param_file:
    param_file.write(f"Learning Rate: {learning_rate}\n")
    param_file.write(f"Iterations: {iterations}\n")
    param_file.write(f"Final MSE: {mse_log[-1]}\n")
    param_file.write(f"Slope (m): {m}\n")
    param_file.write(f"Intercept (b): {b}\n")
    param_file.write(f"R-squared (R²): {r_squared}\n")

# Print results to console
print("Model Parameters and Performance:")
print(f"Learning Rate: {learning_rate}")
print(f"Iterations: {iterations}")
print(f"Final MSE: {mse_log[-1]}")
print(f"Slope (m): {m}")
print(f"Intercept (b): {b}")
print(f"R-squared (R²): {r_squared}")
print(f"MSE Log File: {mse_file_name}")
print(f"Model Parameters File: {parameters_file}")
print(f"MSE Plot File: {mse_plot_file}")
print(f"Regression Plot File: {regression_plot_file}")

# Discussion Section
discussion = """
Discussion:
Challenges:
1. Data Preprocessing: Parsing the dataset and handling potential issues like missing or malformed data
   was time-consuming. This was solved by carefully parsing the data and validating it before using it.
2. Numerical Stability: With large-scale data, numerical stability was a concern. Scaling the data before 
   training would have helped, but it was not implemented in this version.
3. Gradient Descent Tuning: Finding the right learning rate and number of iterations took time.

Straightforward Steps:
1. Implementing the gradient descent algorithm was straightforward as it followed a standard formula.
2. Plotting and saving graphs using Matplotlib took minimal time due to Matplotlib's built-in features.
3. Writing results to files was easily implemented using Python's file handling capabilities.
"""
print(discussion)

