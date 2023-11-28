from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from lags import create_df_with_lags


def perform_random_forest(data):
    df = data["df"].copy()
    if data.get("lags", 0) > 0:
        df = create_df_with_lags(df, data["lags"])
    fecha_corte = data.get("fecha_corte", "2023-07-01")
    X_train = df[df["Date"] < fecha_corte].drop(["Date", "energy_price"], axis=1)
    X_test = df[df["Date"] >= fecha_corte].drop(["Date", "energy_price"], axis=1)
    y_train = df[df["Date"] < fecha_corte]["energy_price"]
    y_test = df[df["Date"] >= fecha_corte]["energy_price"]
    param_dist = data.get(
        "param_dist",
        {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [None, 10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        },
    )
    rf_regressor = RandomForestRegressor()

    random_search = RandomizedSearchCV(
        rf_regressor,
        param_distributions=param_dist,
        n_iter=100,
        cv=5,
        scoring="neg_mean_squared_error",
        random_state=42,
    )

    # Fitting the model
    random_search.fit(X_train, y_train)

    print(f"Best parameters: {random_search.best_params_}")

    best_rf_regressor = random_search.best_estimator_

    y_pred = best_rf_regressor.predict(X_test)

    mse_test = mean_squared_error(y_test, y_pred)

    print("Mean Squared Error on Test Data:", mse_test)
