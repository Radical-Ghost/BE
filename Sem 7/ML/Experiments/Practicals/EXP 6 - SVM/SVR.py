import numpy as np
import pandas as pd 
import plotly.express as px
import seaborn as sns
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, r2_score, root_mean_squared_error, recall_score, precision_score,mean_absolute_error

df = sns.load_dataset('tips')

num_features = df.select_dtypes(include=np.number).columns.tolist()

X = df.drop(columns=['total_bill'])
y = df['total_bill']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23, random_state=42)

le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()

X_train['sex'] = le1.fit_transform(X_train['sex'])
X_train['smoker'] = le2.fit_transform(X_train['smoker'])
X_train['time'] = le3.fit_transform(X_train['time'])

X_test['sex'] = le1.transform(X_test['sex'])
X_test['smoker'] = le2.transform(X_test['smoker'])
X_test['time'] = le3.transform(X_test['time'])

ct = ColumnTransformer(transformers=[('onehot', OneHotEncoder(drop='first'), [3])], remainder='passthrough')

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

svr = SVR()

svr.fit(X_train, y_train)

y_pred = svr.predict(X_test)

print("\nR2 Score:", r2_score(y_test, y_pred))
print("\nRoot Mean Squared Error:", root_mean_squared_error(y_test, y_pred))
print("\nMean Absolute Error:", mean_absolute_error(y_test, y_pred))

fig = px.scatter(
    x=y_test,
    y=y_pred,
    labels={'x': 'Actual Total Bill', 'y': 'Predicted Total Bill'},
    title='Actual vs. Predicted Total Bill'
)

# Add a red line for perfect predictions (y=x)
fig.add_shape(
    type='line',
    x0=y_test.min(), y0=y_test.min(),
    x1=y_test.max(), y1=y_test.max(),
    line=dict(color='Red', dash='dash')
)

fig.show()