# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

#This example uses SVM to classify the Iris dataset into three species of flowers: Setosa, Versicolour, and Virginica.


# Load the Iris dataset and use only the first two features for simplicity
iris = datasets.load_iris()
X = iris.data[:, :2]  # Selecting only the first two features for visualization
y = iris.target  # Target labels (species of iris)

# Split the dataset into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with scaling and SVM classifier
# StandardScaler is used to scale the features, which is often useful for SVMs
# SVC is the Support Vector Classifier
model = make_pipeline(StandardScaler(), SVC(kernel='linear', C=1.0))

# Train the model on the training data
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on the Iris dataset:", accuracy)

# Plotting decision regions (for visualization, using first two features)
# Create a mesh grid to plot decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# Plot decision boundaries
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])  # Predict over the grid
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.2)
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title("SVM Decision Boundary on Iris Dataset (Using First Two Features)")
plt.show()
