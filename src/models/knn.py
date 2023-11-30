from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from lags import create_df_with_lags


def perform_knn(data):
    df = data["df"].copy()
    if data.get("lags", 0) > 0:
        df = create_df_with_lags(df, data["lags"])
    fecha_corte = data.get("fecha_corte", "2023-07-01")
    X_train = df[df["Date"] < fecha_corte].drop(["Date", "energy_price"], axis=1)
    X_train_dates = df[df["Date"] < fecha_corte]["Date"]
    X_test = df[df["Date"] >= fecha_corte].drop(["Date", "energy_price"], axis=1)
    X_test_dates = df[df["Date"] >= fecha_corte]["Date"]
    y_train = df[df["Date"] < fecha_corte]["energy_price"]
    y_test = df[df["Date"] >= fecha_corte]["energy_price"]
    param_dist = data.get(
        "params_dist",
        {
            "n_neighbors": range(1, 30),
            "weights": ["uniform", "distance"],
            "p": [1, 2],
        },
    )
    # Creating the RandomizedSearchCV object
    knn_regressor = KNeighborsRegressor()
    random_search = RandomizedSearchCV(
        knn_regressor,
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )

    # Fitting the model
    random_search.fit(X_train, y_train)

    print(f"Best parameters: {random_search.best_params_}")

    best_knn_regressor = random_search.best_estimator_
    y_pred = best_knn_regressor.predict(X_test)
    y_pred_train = best_knn_regressor.predict(X_train)

    mse_test = mean_squared_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    print("Mean Squared Error on Test Data:", mse_test)
    print(f"Mean Absolute Percentage Error: {mape}")

    return {
        "mape": mape,
        "mse_test": mse_test,
        "X_train_dates": X_train_dates,
        "y_train": y_train,
        "y_pred_train": y_pred_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test_dates": X_test_dates,
    }
