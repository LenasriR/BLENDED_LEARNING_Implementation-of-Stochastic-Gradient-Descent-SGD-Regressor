# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import required libraries and load the dataset by separating input features and target values. 2.Split the dataset into training and testing sets. 3.Create and train the SGD Regressor model using the training data. 4.Predict the output for test data and evaluate the model performance using error metrics.

## Program:
```
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("CarPrice_Assignment.csv")
print(data.head())
print(data.info())
data = data.drop(['CarName', 'car_ID'], axis=1)
data = pd.get_dummies(data, drop_first=True)
X = data.drop('price', axis=1)
y = data['price']
scaler = StandardScaler()
X= scaler.fit_transform(X)
y= scaler.fit_transform(np.array(y).reshape(-1, 1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)
sgd_model.fit(X_train, y_train)
y_pred = sgd_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print('Name: Lenasri R ')

print('Reg. No:212225040199')

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
print("\nModel Coefficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Prices")

plt.ylabel("Predicted Prices")

plt.title("Actual vs Predicted Prices using SGD Regressor")

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')

plt.show()

```

## Output:
<img width="940" height="323" alt="image" src="https://github.com/user-attachments/assets/6546992a-538a-47f7-8927-ca5810d09238" />
<img width="714" height="578" alt="image" src="https://github.com/user-attachments/assets/4264e71f-5c37-406d-be44-56516047b278" />


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
