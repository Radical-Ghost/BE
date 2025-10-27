import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def perform_linear_regression():
    """
    Performs a linear regression analysis on the 'GRE Score' and 'Chance of Admit'
    columns of the Admission_Predict.csv dataset.

    Args:
        file_path (str): The path to the CSV dataset.
    """

    # 1. Setup and Data Loading
    print("--- 1. Setup and Data Loading ---")
    df1 = pd.read_csv("C:\Academic\BE\Sem 7\ML\Practicals\EXP 2 - Linear Regression\Admission_Predict.csv")
    print("Data loaded successfully.")
    print("\n" + "="*50 + "\n")

    # 2. Data Inspection and Preparation
    print("--- 2. Data Inspection and Preparation ---")
    print("First 5 rows of the original dataset:")
    print(df1.head())
    print("\n")

    print("DataFrame Info:")
    df1.info()
    print("\n")

    print("Selecting 'GRE Score' as X and 'Chance of Admit' as y.")
    df = df1[['GRE Score', 'Chance of Admit ']]
    X = df[['GRE Score']]
    y = df['Chance of Admit ']
    print("Selected data preview:")
    print(df.head())
    print("\n" + "="*50 + "\n")

    # 3. Initial Data Visualization
    print("--- 3. Initial Data Visualization ---")
    print("Generating scatter plot of GRE Score vs Chance of Admit (Actual Data)...")
    plt.figure(figsize=(8, 6))
    plt.scatter(X, y, color='blue', label='Actual')
    plt.xlabel('GRE Score')
    plt.ylabel('Chance of Admit')
    plt.title('GRE Score vs Chance of Admit (Actual Data)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("The scatter plot generally suggests a positive correlation, indicating that as GRE scores increase, the chance of admit also tends to increase.")
    print("\n" + "="*50 + "\n")

    # 4. Model Training
    print("--- 4. Model Training ---")
    print("Splitting data into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print("\n")

    print("Initializing and training the Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")
    print("\n" + "="*50 + "\n")

    # 5. Model Evaluation and Prediction
    print("--- 5. Model Evaluation and Prediction ---")
    y_pred = model.predict(X_test)
    print(f"Model Coefficients (Slope): {model.coef_[0]:.6f}")
    print(f"Model Intercept: {model.intercept_:.6f}")
    print("\n")
    print(f"Interpretation: For every one-point increase in GRE Score, the 'Chance of Admit' is predicted to increase by approximately {model.coef_[0]:.4f} (or {model.coef_[0]*100:.2f}%).")
    print("The intercept represents the predicted 'Chance of Admit' when the GRE Score is zero, which is primarily a mathematical component of the line in this context.")
    print("\n" + "="*50 + "\n")

    # 6. Visualizing the Regression Line
    print("--- 6. Visualizing the Regression Line ---")
    print("Generating scatter plot with the fitted linear regression line...")
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.plot(X_test, y_pred, color='red', label='Predicted Regression Line')
    plt.xlabel('GRE Score')
    plt.ylabel('Chance of Admit')
    plt.title('Linear Regression: GRE vs Chance of Admit (Test Data with Prediction)')
    plt.legend()
    plt.grid(True)
    plt.show()
    print("The plot visually demonstrates the fitted linear regression line (in red) against the actual 'Chance of Admit' values from the test set (blue dots), illustrating the learned linear relationship.")
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # Ensure 'Admission_Predict.csv' is in the same directory as this script,
    # or provide the full path to the file.
    perform_linear_regression()