# Logistic Regression

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data
df = pd.read_csv('data.csv')
X = df.iloc[:, 0].values.reshape(-1, 1)
y = df.iloc[:, 1].values

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train model
model = LogisticRegression()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Print results
print("Coefficient (slope):", model.coef_[0][0])
print("Intercept:", model.intercept_[0])
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Plot
plt.scatter(X, y, color='blue', label='Actual data')

# Generate smooth X range for sigmoid curve
x_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_prob = model.predict_proba(x_range)[:, 1]

plt.plot(x_range, y_prob, color='red', linewidth=2, label='Sigmoid curve')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title("Logistic Regression Curve")
plt.legend()
plt.grid(True)
plt.show()
