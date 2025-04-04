# Linear Regression

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load data
df = pd.read_csv('data.csv') 
X = df.iloc[:, 0].values.reshape(-1, 1)  
y = df.iloc[:, 1].values                 

# Split data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Print results
print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)
print("R² Score (Accuracy):", r2_score(y_test, y_pred))

# Plot
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression line')
plt.xlabel(df.columns[0])
plt.ylabel(df.columns[1])
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.show()




