import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Step 1: Generate Mock Data
np.random.seed(0)  # Set seed for reproducibility
X = np.random.rand(100, 1) * 10  # Generate 100 random points (feature values between 0 and 10)
y = (X > 5).astype(int).ravel()  # Create labels: 1 if the feature is greater than 5, else 0
# This simulates a binary classification problem where the threshold is 5.

# Step 2: Split Data into Training and Test Sets
# Split the data into 67% training and 33% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Step 3: Initialize the Logistic Regression Model
log_reg = LogisticRegression()  # Create an instance of Logistic Regression

# Step 4: Fit the Model with Training Data
log_reg.fit(X_train, y_train)  # Train the model using the training data (X_train, y_train)

# Step 5: Predict Probabilities on the Test Data
# predict_proba returns the probability estimates for each class (0 or 1).
# The second column ([:, 1]) represents the probabilities of class 1 (i.e., y=1).
probs = log_reg.predict_proba(X_test)  # Get probabilities for the test data

# Step 6: Evaluate the Model by Calculating the Accuracy
# The score function returns the accuracy of the model: the proportion of correct predictions.
score = log_reg.score(X_test, y_test)

# Step 7: Visualize the Logistic Regression Model
# Plot the test data points
plt.scatter(X_test, y_test, color='red', label='Test Data')  # Red points are the actual test data points

# Step 8: Plot the Logistic Regression Curve
# Generate a smooth range of X values to plot the logistic regression curve
X_range = np.linspace(0, 10, 300).reshape(-1, 1)  # Create 300 points between 0 and 10
y_prob = log_reg.predict_proba(X_range)[:, 1]  # Predict the probability for class 1 over the entire range

# Plot the logistic regression curve (probability of class 1 as a function of X)
plt.plot(X_range, y_prob, color='blue', linewidth=2, label='Logistic Regression Model')

# Step 9: Add Labels and Titles to the Plot
plt.title(f'Logistic Regression Model (Score: {score:.2f})')  # Include the model score in the title
plt.xlabel('X')  # X-axis label
plt.ylabel('Probability')  # Y-axis label (probability of class 1)
plt.legend()  # Show legend
plt.grid(True)  # Show grid for better visualization

# Step 10: Display the Plot
plt.show()  # Display the final plot
