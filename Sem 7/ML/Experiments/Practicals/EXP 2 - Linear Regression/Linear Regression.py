import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Admission_Predict.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Features and target
X = df[['GRE Score']]
y = df['Chance of Admit']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Output coefficients
print(f"Model Coefficients: {model.coef_}, Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# Plot 1: Original GRE vs Chance of Admit
axes[0].scatter(X, y, color='blue', label='Actual')
axes[0].set_title('Train Data: GRE Score vs Chance of Admit')
axes[0].set_xlabel('GRE Score')
axes[0].set_ylabel('Chance of Admit')
axes[0].legend()

# Plot 2: Predictions
axes[1].scatter(X_test, y_test, color='blue', label='Actual')
axes[1].plot(X_test, y_pred, color='red', label='Predicted')
axes[1].set_title('Test Data: Prediction vs Actual')
axes[1].set_xlabel('GRE Score')
axes[1].set_ylabel('Chance of Admit')
axes[1].legend()

# Adjust layout and show
plt.tight_layout()
plt.show()