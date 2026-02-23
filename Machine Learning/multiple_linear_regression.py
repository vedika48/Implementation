import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics

data_url = "https://lib.stat.cmu.edu/datasets/boston"
raw = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)

x = np.hstack([raw.values[::2, :], raw.values[1::2, :2]])
y = raw.values[1::2, 2]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

#Model tranining
regression = linear_model.LinearRegression()
regression.fit(x_train, y_train)

#Prediction
y_pred = regression.predict(x_test)

# Coefficient 
print('Coefficient: ', regression.coef_)
print('Intercept: ', regression.intercept_)

# Model evaluation
print("Mean absolute error: ", metrics.mean_absolute_error(y_test, y_pred))
print("Mean squared error: ", metrics.mean_squared_error(y_test, y_pred))
print("R² Score:", metrics.r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()