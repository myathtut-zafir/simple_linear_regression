import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from pandas.core.common import random_state
from sklearn.linear_model import LinearRegression

salary_data=pd.read_csv('./data/Salary_Data.csv')
# print(salary_data.head())

# plt.title("salary")
# sns.displot(salary_data['Salary'])
# plt.show()

# plt.scatter(salary_data['YearsExperience'], salary_data['Salary'], color = 'lightcoral')
# plt.title('Salary vs Experience')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.box(False)
# plt.show()

# Splitting variables
X = salary_data.iloc[:, :1]  # independent
y = salary_data.iloc[:, 1:]  # dependent
# print(y)

# Splitting dataset into test/train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Regressor model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction result
y_pred_test = regressor.predict(X_test)     # predicted value of y_test
y_pred_train = regressor.predict(X_train)   # predicted value of y_train

# print(y_pred_train)
print(y_pred_test)