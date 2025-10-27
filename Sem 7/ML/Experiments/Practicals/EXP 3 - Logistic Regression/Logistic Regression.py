import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, root_mean_squared_error, recall_score, precision_score

df = pd.read_csv('C:\Academic\BE\Sem 7\ML\Practicals\EXP 3 - Logistic Regression\Social_Network_Ads.csv')
df = df.drop(columns=['User ID'])
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def print_dataset():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Training set
    sns.scatterplot(
        x=X_train['Age'], 
        y=X_train['EstimatedSalary'],
        hue=y_train,
        style=X_train['Gender'],
        palette='mako', 
        s=50, ax=axes[0]
    )
    axes[0].set_title("Training Set")

    # Test set
    sns.scatterplot(
        x=X_test['Age'], 
        y=X_test['EstimatedSalary'],
        hue=y_test,
        style=X_test['Gender'],
        palette='mako', 
        s=50, ax=axes[1]
    )
    axes[1].set_title("Test Set")

    plt.suptitle("Age vs Salary split by Purchased & Gender")
    plt.tight_layout()
    plt.show()

print_dataset()

model = LogisticRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(f"Model Coefficients: {model.coef_}, Intercept: {model.intercept_}")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nR2 Score:", r2_score(y_test, y_pred))
print("\nRoot Mean Squared Error:", root_mean_squared_error(y_test, y_pred))
print("\nRecall:", recall_score(y_test, y_pred))
print("\nPrecision:", precision_score(y_test, y_pred))

# def scatter_actual_vs_predicted(y_test, y_pred, X_test):
#     plt.figure(figsize=(8, 6))
#     ax = plt.gca()

#     sns.scatterplot(
#         x=X_test['Age'],
#         y=X_test['EstimatedSalary'],
#         hue=y_pred,
#         style=X_test['Gender'],
#         palette='mako',
#         markers={0: 'X', 1: 'X'},
#         s=150,
#         ax=ax,
#         legend=False
#     )

#     sns.scatterplot(
#         x=X_test['Age'],
#         y=X_test['EstimatedSalary'],
#         hue=y_test,
#         style=X_test['Gender'],
#         palette='mako',
#         markers={0: 'o', 1: 'o'},
#         s=50,
#         ax=ax,
#         legend=False
#     )

#     custom_legend = [
#         Line2D([0], [0], marker='o', color='w', label='Actual', markerfacecolor='gray', markeredgecolor='black', markersize=8),
#         Line2D([0], [0], marker='X', color='w', label='Predicted', markerfacecolor='gray', markeredgecolor='black', markersize=8),
#         Line2D([0], [0], marker='s', color='w', label='Not Purchased (0)', markerfacecolor='#40498e', markersize=10),
#         Line2D([0], [0], marker='s', color='w', label='Purchased (1)', markerfacecolor='#38aaac', markersize=10)
#     ]

#     plt.legend(handles=custom_legend, title='Legend', loc='upper right')
#     plt.title("Actual vs Predicted - Purchased Classification")
#     plt.xlabel("Age")
#     plt.ylabel("Estimated Salary")
#     plt.grid(True)
#     plt.show()
# scatter_actual_vs_predicted(y_test, y_pred, X_test)

def plot_decision_boundaries_side_by_side(model, scaler):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    for i, gender_value in enumerate([0, 1]):
        # Create meshgrid
        age_range = np.linspace(X['Age'].min(), X['Age'].max(), 200)
        salary_range = np.linspace(X['EstimatedSalary'].min(), X['EstimatedSalary'].max(), 200)
        xx, yy = np.meshgrid(age_range, salary_range)
        grid = np.c_[np.full(xx.ravel().shape, gender_value), xx.ravel(), yy.ravel()]
        grid_scaled = scaler.transform(grid)
        Z = model.predict(grid_scaled).reshape(xx.shape)

        # Create DataFrame for test points
        plot_df = X_test.copy()
        plot_df['Actual'] = y_test.values
        plot_df['Predicted'] = y_pred
        plot_df = plot_df[plot_df['Gender'] == gender_value]

        ax = axes[i]
        ax.set_title(f'Decision Boundary (Gender={gender_value})')
        ax.set_xlabel('Age')
        if i == 0:
            ax.set_ylabel('Estimated Salary')

        # Decision surface
        ax.contourf(xx, yy, Z, alpha=0.3, cmap='mako')

        # Test points
        sns.scatterplot(
            data=plot_df,
            x='Age', y='EstimatedSalary',
            hue='Actual',
            style='Predicted',
            palette='mako',
            edgecolor='black',
            s=70,
            ax=ax,
            legend=False
        )

    # Custom legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker='o', color='w', label='Actual', markerfacecolor='black', markersize=8),
        Line2D([0], [0], marker='X', color='w', label='Predicted', markerfacecolor='black', markersize=8),
        Line2D([0], [0], marker='s', color='w', label='Not Purchased (0)', markerfacecolor='#40498e', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Purchased (1)', markerfacecolor='#38aaac', markersize=10)
    ]
    fig.legend(handles=handles, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), title="Legend")

    plt.tight_layout()
    plt.show()

plot_decision_boundaries_side_by_side(model, scaler)