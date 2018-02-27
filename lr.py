from zipfile import ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns

bikes = pd.read_csv('bsd/hour.csv', index_col='dteday', parse_dates=True)

bikes['hour'] = bikes.index.hour

bikes.head()
bikes.tail()

# - **hour** ranges from 0 (midnight) through 23 (11pm)
# - **workingday** is either 0 (weekend or holiday) or 1 (non-holiday weekday)

# ## Task 1
#
# Run these two `groupby` statements and figure out what they tell you about the data.

# mean rentals for each value of "workingday"
# sns.set(style='whitegrid', context='notebook')
# cols = ['hr', 'temp', 'atemp', 'hum', 'windspeed', 'cnt']
# sns.pairplot(bikes[cols], size=2.5)
# plt.show()


#WARNING: dont run code below(mem overflow)
# cm = np.corrcoef(bikes[cols])
#
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size':15}, yticklabels=cols, xticklabels=cols)
# plt.show()


# bikes.groupby('workingday').cnt.mean()

# mean rentals for each value of "hour"
# bikes.groupby('hour').cnt.mean()

# bikes.groupby(['holiday', 'season']).cnt.mean().unstack().plot()

feature_cols = ['casual']

X = bikes[feature_cols].values
y = bikes.cnt.values

# X = StandardScaler().fit(X.reshape(-1, 1)).transform(X.reshape(-1, 1))
# y_scaler = StandardScaler().fit(y.reshape(-1, 1))
# y = y_scaler.transform(y.reshape(-1, 1))

X_len = len(X)
test_value = round(X_len * 0.05)

X_train, X_test = X[:-test_value], X[-test_value:]
y_train, y_test = y[:-test_value], y[-test_value:]

linreg = linear_model.LinearRegression()
linreg.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))
y_pred = linreg.predict(X_test.reshape(test_value, 1))

plt.scatter(X_test.reshape(-1,1), y_test, color='b')
plt.plot(X_test.reshape(-1,1), y_pred, color='red',linewidth=1)
plt.show()

# pred = linreg.predict(X_test)
#
# # scores = cross_val_score(linreg, X, y, cv=10, scoring='mean_squared_error')
#
# # The coefficients
print('Coefficients: \n', linreg.coef_)
# # The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# # Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
#
# pass