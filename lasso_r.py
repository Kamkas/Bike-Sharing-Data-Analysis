import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline
import seaborn as sns

bikes = pd.read_csv('bsd/hour.csv', index_col='dteday', parse_dates=True)

bikes['hour'] = bikes.index.hour

bikes.head()
bikes.tail()

# feature_cols = ['temp', 'season', 'weathersit', 'hum']


# multiple scatter plots in Seaborn
# sns.pairplot(bikes, x_vars=feature_cols, y_vars='cnt', kind='reg')
#
#
# # multiple scatter plots in Pandas
# fig, axs = plt.subplots(1, len(feature_cols), sharey=True)
# for index, feature in enumerate(feature_cols):
#     bikes.plot(kind='scatter', x=feature, y='cnt', ax=axs[index], figsize=(16, 3))
#
# plt.show()

"""
    Lasso regression
"""

feature_cols = ['hr']


X = bikes[feature_cols]
y = bikes.cnt

X = StandardScaler().fit(X.values.reshape(-1,1)).transform(X.values.reshape(-1,1))
y = StandardScaler().fit(y.values.reshape(-1,1)).transform(y.values.reshape(-1,1))

X_len = len(X)
test_value = round(X_len * 0.1)

X_train, X_test = X[:-test_value], X[-test_value:]
y_train, y_test = y[:-test_value], y[-test_value:]

degrees = list(range(12,17))

def MakeExample(index, plt, model):
    # create polynomic coefs (features) for polynomic regression
    polynomial_features = PolynomialFeatures(degree=degrees[index],
                                             include_bias=False)

    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", model)])
    pipeline.fit(X_train, y_train)

    # Evaluate the models using crossvalidation
    scores = cross_val_score(pipeline, X_train, y_train,
                             scoring="neg_mean_squared_error", cv=10)

    # X_test = np.linspace(0, 1, 100)
    # visualization source data (points)
    y_pred = pipeline.predict(X_test)
    plt.plot(X_test, y_pred, label="Model")

    plt.scatter(X, y, color='lightgray', s=4, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.xlim((-1, 24))
    # plt.ylim((0, 1000))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
    print('For X^{0}'.format(degrees[i]))
    print('R2 score {0}'.format(r2_score(y_test, y_pred)))
    print("Degree {}\nMSE = {:.4e}(+/- {:.4e})".format(
        degrees[i], -scores.mean(), scores.std()))
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))

    return plt

plt.figure(figsize=(40,20))

for i in range(len(degrees)):
    ridge = linear_model.Lasso(alpha=10**-5)
    plt.subplot(2, len(degrees), i+1)
    plt = MakeExample(i, plt, ridge)



plt.show()