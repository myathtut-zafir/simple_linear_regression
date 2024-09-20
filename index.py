import numpy as np
from sklearn.linear_model import LinearRegression

# Features
X = np.array([
    [10, 100],  # house 1
    [15, 200],  # house 2
])
#Label
y = np.array([1000, 1500])
model=LinearRegression()
model.fit(X,y)

house_features = np.array([[20, 300]])
predicted_price = model.predict(house_features)
print(predicted_price)