# Step 1: Import necessary libraries
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression  # Using Logistic Regression as an example model
from sklearn.datasets import load_iris
import numpy as np

# Step 2: Load your dataset
# For demonstration, using Iris dataset which is a multiclass classification problem
data = load_iris()
X = data.data
y = data.target

# Step 3: Create a Logistic Regression classifier
# Logistic Regression is used here as an example; you can use any classifier depending on your problem
model = LogisticRegression(solver='liblinear', multi_class='ovr')

# Step 4: Define the k-fold cross-validation configuration
# StratifiedKFold is used to ensure each fold of dataset has the same proportion of observations with a given label
skf = StratifiedKFold(n_splits=5)  # Setting number of splits/folds to 5

# Step 5: Execute the cross-validation
# cross_val_score runs the model using the cross-validation approach defined by skf
# It automatically uses the StratifiedKFold for splitting the data if the CV parameter is an integer
scores = cross_val_score(model, X, y, scoring='accuracy', cv=skf)

# Step 6: Print the outputs
print("Accuracy scores for each fold are:", scores)
print("Mean accuracy score:", np.mean(scores))
