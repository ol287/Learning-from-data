import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(0)

# Sample data: X is a single feature, y is the corresponding target variable
X = 2 * np.random.rand(100, 1)  # 100 samples, 1 feature
y = 4 + 3 * X[:, 0] + np.random.randn(100)  # Linear relation y = 4 + 3X plus noise

# Reshape X for compatibility with scikit-learn
X = X.reshape(-1, 1)


# Using Ridge Regression
ridge_regression = linear_model.Ridge(alpha=1.0)
ridge_regression.fit(X, y)

# Using Lasso Regression
lasso_regression = linear_model.Lasso(alpha=1.0)
lasso_regression.fit(X, y)

# Output the coefficients from each model
print("Ridge Regression Coefficients:", ridge_regression.coef_)
print("Lasso Regression Coefficients:", lasso_regression.coef_)

print("Lasso Regression (Least Absolute Shrinkage and Selection Operator): Lasso regression is a type of linear regression that uses shrinkage. It adds a penalty equal to the absolute value of the magnitude of coefficients to the loss function. This regularization can lead to sparse models where some coefficients are exactly zero, effectively selecting more significant features and discarding the less significant ones.")
print("Ridge Regression: Ridge regression is a method of linear regression where a small, squared penalty term is added to the loss function to prevent overfitting. This penalty is proportional to the square of the magnitude of the coefficients, encouraging them to be small but typically not zero. This method is particularly useful when dealing with multicollinearity or when the number of predictors exceeds the number of observations.")

# Ridge Regression Plot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, ridge_regression.predict(X), color='red', label='Ridge Model')
plt.title('Ridge Regression Fit')
plt.xlabel('Feature X')
plt.ylabel('Target y')
plt.legend()

# Lasso Regression Plot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, lasso_regression.predict(X), color='green', label='Lasso Model')
plt.title('Lasso Regression Fit')
plt.xlabel('Feature X')
plt.ylabel('Target y')
plt.legend()

plt.tight_layout()
plt.show()
