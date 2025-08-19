# Project 1: Individual 6
# Full Name: Issa Tijani
# Class Name: DATA 527
# Deadline: 3/1/2025
# Term: Spring 2025


import matplotlib.pyplot as plt
import openpyxl
import math

while True:
    # Get user input for hyperparameters
    learning_rate = input("Enter learning rate (or press 'x' to exit): ")
    if learning_rate.lower() == 'x':
        break
    iterations = input("Enter number of iterations (or press 'x' to exit): ")
    if iterations.lower() == 'x':
        break

    try:
        learning_rate = float(learning_rate)
        iterations = int(iterations)
    except ValueError:
        print("Invalid input. Please enter numeric values.")
        continue


    # Load dataset from .xlsx file and filter out rows with missing values
    def load_data():
        wb = openpyxl.load_workbook("myDataMLR.xlsx")
        sheet = wb.active
        X, y = [], []
        for row in sheet.iter_rows(min_row=2, values_only=True):
            if None in row:  # Skip rows with missing values
                continue
            values = [float(cell) for cell in row]
            X.append(values[:-1])  # Independent variables
            y.append(values[-1])  # Dependent variable
        return X, y


    # if the dataset is empty, restart the loop
    X, y = load_data()
    if not X or not y:
        print("No complete data available in the file.")
        continue

    n_features = len(X[0])
    n_samples = len(y)


    # Normalize the features and target variable
    def normalize(values):
        mean = sum(values) / len(values)
        stddev = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        return [(x - mean) / stddev for x in values], mean, stddev


    X = [normalize([row[j] for row in X])[0] for j in range(n_features)] # each feature is normalized
    y, y_mean, y_std = normalize(y)

    # Transpose X for easier access
    X = list(map(list, zip(*X)))

    # Initialize parameters
    coefficients = [0] * n_features
    intercept = 0
    mse_log = []

    # Gradient Descent
    for i in range(iterations):
        predictions = [sum(coefficients[j] * X[i][j] for j in range(n_features)) + intercept for i in range(n_samples)]
        errors = [predictions[i] - y[i] for i in range(n_samples)]
        mse = sum(e ** 2 for e in errors) / (2 * n_samples)
        mse_log.append(mse)

        gradient_intercept = sum(errors) / n_samples
        gradients = [sum(errors[i] * X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]

        intercept -= learning_rate * gradient_intercept
        for j in range(n_features):
            coefficients[j] -= learning_rate * gradients[j]

    # Compute R-squared and RMSE
    y = [(yi * y_std) + y_mean for yi in y] # Denormalize y
    predictions = [(p * y_std) + y_mean for p in predictions] # Denormalize prediction
    y_mean = sum(y) / n_samples
    ss_total = sum((yi - y_mean) ** 2 for yi in y)
    ss_residual = sum((y[i] - predictions[i]) ** 2 for i in range(n_samples))
    r2 = 1 - (ss_residual / ss_total)
    rmse = math.sqrt(ss_residual / n_samples)

    # Save MSE for each iteration to log file
    with open(f"MLRTraining[{iterations}][{learning_rate}]MSE.log", "w") as f:
        for value in mse_log:
            f.write(f"{value}\n")
   # Save final model parameters, performance metrics and a results to a separate log file
    with open("MLRModelParameters.log", "a") as f:
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Iterations: {iterations}\n")
        f.write(f"Final MSE: {mse}\n")
        f.write(f"Intercept: {intercept}\n")
        f.write(f"Coefficients: {coefficients}\n")
        f.write(f"R-squared: {r2}\n")
        f.write(f"RMSE: {rmse}\n")
        f.write("------------------------------------------\n")

    # Print results in table format
    print("\nResults Table:")
    print(f"{'Parameter':<20}{'Value'}")
    print("-" * 40)
    print(f"{'Learning Rate':<20}{learning_rate}")
    print(f"{'Iterations':<20}{iterations}")
    print(f"{'Final MSE':<20}{mse}")
    print(f"{'Intercept':<20}{intercept}")
    print(f"{'Coefficients':<20}{coefficients}")
    print(f"{'R-squared':<20}{r2}")
    print(f"{'RMSE':<20}{rmse}")
    print("-" * 40)

    # Save table to the same log file
    with open("MLROutputTable.log", "a") as f:
        f.write("\nResults Table:\n")
        f.write(f"{'Parameter':<20}{'Value'}\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Learning Rate':<20}{learning_rate}\n")
        f.write(f"{'Iterations':<20}{iterations}\n")
        f.write(f"{'Final MSE':<20}{mse}\n")
        f.write(f"{'Intercept':<20}{intercept}\n")
        f.write(f"{'Coefficients':<20}{coefficients}\n")
        f.write(f"{'R-squared':<20}{r2}\n")
        f.write(f"{'RMSE':<20}{rmse}\n")
        f.write("-" * 40 + "\n")

    # Plot actual vs predicted values with error lines
    plt.scatter(y, predictions, label="Predicted vs Actual")
    for i in range(len(y)):
        plt.plot([y[i], y[i]], [y[i], predictions[i]], 'r-', label="Error" if i == 0 else "")  # Error lines
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted Values")
    plt.legend()
    plt.savefig("Actual_vs_Predicted.png")
    plt.show()

    # Plot MSE per iteration
    plt.plot(range(iterations), mse_log, label="MSE")
    plt.xlabel("Iterations")
    plt.ylabel("Mean Squared Error")
    plt.title("MSE per Iteration")
    plt.legend()
    plt.savefig("MSE_per_Iteration.png")
    plt.show()
