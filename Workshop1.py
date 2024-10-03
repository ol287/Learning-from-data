import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

#Question 1
cars = pd.read_csv("/content/sample_data/cars_dataset.csv")

#Question 2
print(cars.head())

#Question 3
plt.figure(figsize=(8,6))
plt.scatter(cars['weight'], cars['horsepower'], alpha=0.6)
plt.title('Scatter Plot of Weight vs Horsepower')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.grid(True)
plt.show()

#Question 4
# Define the features (X) and target (y)
X = cars[['weight']]  # Independent variable (reshape if necessary)
y = cars['horsepower']  # Dependent variable

#Question 5
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Predict horsepower from weight
y_pred = model.predict(X)


# Plot the scatter plot and regression line
plt.figure(figsize=(8,6))
plt.scatter(cars['weight'], cars['horsepower'], alpha=0.6, label='Actual data')
plt.plot(cars['weight'], y_pred, color='red', label='Regression line')
plt.title('Scatter Plot with Linear Regression: Weight vs Horsepower')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.legend()
plt.grid(True)
plt.show()

#Question 6
# Print model coefficients
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")

#Question 7
#first 10 predicted values
y_pred[:10]

#Question 8 
cars["predicted_horsepower"]= y_pred

#Question 9
ax = cars.plot.scatter(x='weight', y='horsepower', color='blue', label='Actual horsepower')
cars.plot.line(x='weight', y='predicted_horsepower', color='red', ax=ax, label='Predicted horsepower')

# Titles and labels
plt.title('Regression Analysis: Actual vs Predicted Horsepower')
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.legend()
plt.grid(True)
plt.show()

#Question 10 
r_squared = model.score(X, y)

# Print the R-squared value
print(f"R-squared value: {r_squared:.2f}")
