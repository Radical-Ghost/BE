import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, root_mean_squared_error, recall_score, precision_score, mean_absolute_error

df = pd.read_csv('./car_evaluation.csv')
col_names = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df.columns = col_names

X = df.drop(['class'], axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify=y) 

encoder = ce.OrdinalEncoder(cols=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)

parms = {
    'criterion': ['gini', 'log_loss', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [6, 7, 8],
}

model = DecisionTreeClassifier()
grid = GridSearchCV(model, param_grid=parms, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print("\nBest parameters: ", grid.best_params_)
print("\nBest Score: ", grid.best_score_)

y_pred = grid.predict(X_test)

plt.figure(figsize=(32, 20))
tree.plot_tree(grid.best_estimator_, filled=True)
plt.show()

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))