# Import necessary libraries
import numpy as np  # NumPy for numerical operations and array handling
from sklearn.linear_model import LinearRegression  # Import Linear Regression model from scikit-learn
from sklearn.model_selection import train_test_split  # Tool to split data into training and testing sets
import matplotlib.pyplot as plt  # Matplotlib for plotting graphs

# Create synthetic data
np.random.seed(0)  # Set random seed for reproducibility
X = 2 * np.random.rand(100, 1)  # Generate 100 random x values, scaled by 2
y = 4 + 3 * X + np.random.randn(100, 1)  # Generate y values based on the equation y = 4 + 3X + noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% training, 20% test

# Create an instance of the Linear Regression model
lr = LinearRegression()  # Create the Linear Regression object

# Fit the model on the training data
lr.fit(X_train, y_train)  # Train the model using the training data

# Predict the responses for the test data
y_pred = lr.predict(X_test)  # Use the trained model to make predictions on the test data

# Plotting the results
plt.scatter(X_test, y_test, color='blue', label='Actual')  # Plot the actual test values
plt.plot(X_test, y_pred, color='red', label='Predicted')  # Plot the predicted values from the model
plt.title('Linear Regression Predictions')  # Title of the plot
plt.xlabel('X values')  # Label for the x-axis
plt.ylabel('y values')  # Label for the y-axis
plt.legend()  # Add a legend to differentiate actual and predicted values
plt.show()  # Display the plot
