import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import seaborn as sns

bikes = pd.read_csv('bsd/hour.csv', index_col='dteday', parse_dates=True)

bikes['hour'] = bikes.index.hour

bikes.head()
bikes.tail()

features_cols = ['hr']

X = bikes[features_cols].values
y = bikes.cnt

X_len = len(X)
test_value = round(X_len * 0.33)

# X = StandardScaler().fit(X.values.reshape(-1, 1)).transform(X.values.reshape(-1, 1))
y_scaler = StandardScaler().fit(y.values.reshape(-1, 1))
y = y_scaler.transform(y.values.reshape(-1, 1))

plt.figure(figsize=(40,20))

degrees = list(range(7,14))

for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i+1)
    plt.setp(ax, xticks=(), yticks=())

    poly_features = PolynomialFeatures(degree=degrees[i])
    linear_regr = linear_model.LinearRegression()

    pipeline = Pipeline([("polynomial_features", poly_features),
                         ("linear_regression", linear_regr)])

    pipeline.fit(X[:-test_value].reshape(-1,1), y[:-test_value])

    scores = cross_val_score(pipeline, X[:-test_value].reshape(-1,1), y[:-test_value], scoring="neg_mean_squared_error", cv=10)

    y_pred = pipeline.predict(X[:-test_value].reshape(-1,1))

    plt.plot(X[:-test_value], y_pred, label="Model degree = {0}".format(degrees[i]))
    plt.scatter(X, y, color='lightgray', s=4, label="Samples")
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.xlim((0,1))
    # plt.ylim((0,1))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))

    print('For X^{0}'.format(degrees[i]))
    print('R2 score {0}'.format(r2_score(y[-test_value:], y_pred[-test_value:])))
    print("Degree {}\nMSE = {:.4e}(+/- {:.4e})".format(
        degrees[i], -scores.mean(), scores.std()))
    print("Mean squared error: %.2f"
          % mean_squared_error(y[-test_value:], y_pred[-test_value:]))




plt.show()