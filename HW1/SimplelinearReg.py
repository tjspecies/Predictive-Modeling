# Full Name: Issa Tijani
# Class Name: DATA 527
# Deadline: 01/29/2025
# Term: Spring 2025

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

# Read data
hours = []
grades = []
with open("StudentGrade.csv", "r") as dataFile:
    lines = dataFile.readlines()
    del lines[0]  # Remove the header

    for line in lines:
        lineToAdd = line.strip().split(",")
        hours.append(float(lineToAdd[0]))  # Convert to float
        grades.append(float(lineToAdd[1]))  # Convert to float

# Perform linear regression
slope, intercept, r_value, p_value, stdDev = scipy.stats.linregress(hours, grades)

# Define function to estimate grades
def estimate_grade(hours_studied):
    return slope * hours_studied + intercept

# Generate estimated grades
estimated_grades = list(map(estimate_grade, hours))

# Compute RMSE
rmse = np.sqrt(np.mean([(obs - pred) ** 2 for obs, pred in zip(grades, estimated_grades)]))

# Print required values
print("Slope:", slope)
print("Intercept:", intercept)
print("RMSE:", rmse)
print("Correlation Coefficient:", r_value)

# Plot the data
plt.scatter(hours, grades, label="Actual Data")
plt.plot(hours, estimated_grades, color='red', label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Grades")
plt.title("Linear Regression: Hours Studied vs Grades")
plt.legend()
plt.savefig("regression_plot.png")  # Save the plot in the current directory
plt.show()

# Prediction Loop
while True:
    try:
        user_input = float(input("Enter the number of hours studied to predict the grade: "))
        predicted_grade = estimate_grade(user_input)
        print(f"Predicted Grade for {user_input} hours of study: {predicted_grade}")

        choice = input("Do you want to predict again? (y/n): ").strip().lower()
        if choice != "y":
            break  # Exit the loop if user enters anything other than 'y'
    except ValueError:
        print("Invalid input. Please enter a valid number.")

print("The code execution has ended!")
print("Bye!")
