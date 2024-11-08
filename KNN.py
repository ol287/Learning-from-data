# Import necessary libraries for data handling, model training, and visualization
from sklearn.datasets import load_wine  # Loads a pre-built dataset of wine characteristics
from sklearn.model_selection import train_test_split  # Helps split data into training and testing sets
from sklearn.neighbors import KNeighborsClassifier  # The K-Nearest Neighbors classifier
from sklearn.metrics import accuracy_score, classification_report  # Metrics to evaluate model performance
import matplotlib.pyplot as plt  # Library for plotting
import numpy as np  # Library for numerical operations

# Load the wine dataset, which is a well-known dataset for classification problems
data = load_wine()
X = data.data  # 'X' contains the features (measurements of each wine sample)
y = data.target  # 'y' contains the target labels (type of wine)

# Split the data into training and test sets
# The training set is used to "teach" the model, while the test set is used to evaluate its performance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set the number of neighbors (k) for KNN. This is the number of closest points the model considers.
k = 5
# Initialize the KNN classifier with our chosen k value
knn = KNeighborsClassifier(n_neighbors=k)

# Train (or "fit") the KNN classifier on the training data
# This step allows the model to "learn" from the training data
knn.fit(X_train, y_train)

# Use the trained model to make predictions on the test data
# This is where the model tries to predict the wine type based on the test features
y_pred = knn.predict(X_test)

# Calculate the accuracy by comparing the model's predictions to the actual labels in the test set
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with k={k}: {accuracy * 100:.2f}%")  # Print the accuracy as a percentage

# Print a detailed report showing precision, recall, and f1-score for each class (wine type)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# Function to plot decision boundaries (only using the first two features for simplicity)
def plot_decision_boundaries(X, y, classifier, title="KNN Decision Boundary"):
    # Create a grid of points that covers the feature space (2D space)
    # This helps visualize where different classes are predicted
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1  # Range for feature 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1  # Range for feature 2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Use the classifier to predict the class of each point in the grid
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)  # Reshape to match the grid shape

    # Plot the decision boundary by coloring different regions for each class
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    # Plot the actual data points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor="k", s=20)
    # Label the axes with the names of the first two features
    plt.xlabel(data.feature_names[0])
    plt.ylabel(data.feature_names[1])
    plt.title(title)  # Set the plot title
    plt.show()  # Display the plot

# For visualization purposes, we reduce the data to only the first two features (since KNN normally uses all features)
X_train_2D = X_train[:, :2]  # Select the first two features in the training set
X_test_2D = X_test[:, :2]    # Select the first two features in the test set
# Initialize a new KNN classifier for the 2D data (with the same k value)
knn_2D = KNeighborsClassifier(n_neighbors=k)
knn_2D.fit(X_train_2D, y_train)  # Train the 2D KNN classifier on the reduced feature set

# Plot the decision boundaries for the 2D feature set
plot_decision_boundaries(X_train_2D, y_train, knn_2D, title=f"KNN Decision Boundary (k={k})")

# Experiment with different values of k to see how the number of neighbors affects accuracy
k_values = range(1, 11)  # Test k values from 1 to 10
accuracies = []  # List to store accuracy scores for each k value

# Loop through each k value, train a new KNN model, and calculate accuracy
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)  # Create a new KNN classifier with current k
    knn.fit(X_train, y_train)  # Train on the training data
    y_pred = knn.predict(X_test)  # Predict on the test data
    accuracies.append(accuracy_score(y_test, y_pred))  # Store the accuracy for this k

# Plot the accuracy for each k value to find the best k for this dataset
plt.figure()  # Create a new figure for the plot
plt.plot(k_values, accuracies, marker='o')  # Plot k values on x-axis and accuracy on y-axis
plt.xlabel("k (Number of Neighbors)")  # Label x-axis
plt.ylabel("Accuracy")  # Label y-axis
plt.title("Accuracy for Different k Values")  # Set plot title
plt.show()  # Display the plot
