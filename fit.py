#!/usr/env/bin python

import numpy as np


def augment_feature(X, X2):
    X_aug = X.copy()
    X2_aug = X2.copy()
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            if i == j:
                continue
            r = X[:, i] / X[:, j]
            r2 = X2[:, i] / X2[:, j]
            if np.isfinite(r).all() and np.isfinite(r2).all():
                X_aug = np.append(X_aug, r[:, np.newaxis], axis=1)
                X2_aug = np.append(X2_aug, r2[:, np.newaxis], axis=1)
    return X_aug, X2_aug


def GPR(X_train, X_test, y_train, y_test):
    import sklearn.gaussian_process as gp

    kernel = gp.kernels.ConstantKernel(1.0, (1e-5, 1e5)) * gp.kernels.RBF(10.0, (1e-5, 1e5))
    model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)
    model.fit(X_train, y_train)
    params = model.kernel_.get_params()

    y_predicted, std = model.predict(X_test, return_std=True)
    return y_predicted


def NN(X_train, X_test, y_train, y_test):
    import keras

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(X_train.shape[1], input_dim=X_train.shape[1], activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.fit(X_train, y_train, batch_size=128, epochs=1000)

    y_predicted = model.predict(X_test)
    return y_predicted


def polynomial(X_train, X_test, y_train, y_test):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression(fit_intercept=True))
    model = model.fit(X_train, y_train)

    y_predicted = model.predict(X_test)
    return y_predicted


def random_forest(X_train, X_test, y_train, y_test):
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    return y_predicted


def SVR(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVR
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.2))
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    return y_predicted


def xgboost(X_train, X_test, y_train, y_test):
    import xgboost

    model = xgboost.XGBRegressor(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    return y_predicted


def xgboost_with_tuning(X_train, X_test, y_train, y_test):
    from sklearn.model_selection import GridSearchCV
    import xgboost

    xgb = xgboost.XGBRegressor(random_state=0)
    xgb_parameters = {'n_estimators': [10, 20, 50, 100, 200, 300],
                      'max_depth': [5, 10, 15, 20],
                      'learning_rate': [1e-2, 0.1, 0.2, 0.5, 0.8],
                      }
    xgb_grid = GridSearchCV(estimator=xgb, param_grid=xgb_parameters, n_jobs=-1)
    xgb_grid_fit = xgb_grid.fit(X_train, y_train)
    model = xgb_grid_fit.best_estimator_
    y_predicted = model.predict(X_test)
    return y_predicted


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    data = np.loadtxt('Concrete_Data.txt')
    X_train, X_test, y_train, y_test = train_test_split(data[:, 0:8], data[:, 8], test_size=0.15, random_state=0)
    X_train_aug, X_test_aug = augment_feature(X_train, X_test)

    methods = {
            '2nd-Order Polynomial': polynomial,
            'Support Vector Regression': SVR,
            'Gaussian Process Regression': GPR,
            #'Neural Networks': NN,
            'Random Forest': random_forest,
            'XGBoost': xgboost,
            }

    for name, func in methods.items():
        y_predicted = func(X_train, X_test, y_train, y_test)
        R2 = r2_score(y_test, y_predicted)
        MSE = mean_squared_error(y_test, y_predicted)

        y_aug_predicted = func(X_train_aug, X_test_aug, y_train, y_test)
        R2_aug = r2_score(y_test, y_aug_predicted)
        MSE_aug = mean_squared_error(y_test, y_aug_predicted)

        print('{}: R2 = {}, MSE = {}, R2(aug) = {}, MSE(aug) = {}'
                .format(name, R2, MSE, R2_aug, MSE_aug))

    import matplotlib.pyplot as plt
    plt.scatter(y_predicted, y_test)
    plt.show()
