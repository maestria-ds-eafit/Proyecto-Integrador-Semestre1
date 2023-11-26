from sklearn.model_selection import RandomizedSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from lags import create_df_with_lags
from sklearn.linear_model import Lasso
import numpy as np


def perform_lasso_mlr(data):
    df = data["df"]
    if data.get("lags", 0) > 0:
        df = create_df_with_lags(df, data["lags"])
    fecha_corte = data.get("fecha_corte", "2023-07-01")
    X_train = df[df["Date"] < fecha_corte].drop(["Date", "energy_price"], axis=1)
    X_train_dates = df[df["Date"] < fecha_corte]["Date"]
    X_test = df[df["Date"] >= fecha_corte].drop(["Date", "energy_price"], axis=1)
    X_test_dates = df[df["Date"] >= fecha_corte]["Date"]
    y_train = df[df["Date"] < fecha_corte]["energy_price"]
    y_test = df[df["Date"] >= fecha_corte]["energy_price"]
    lasso = Lasso()

    # Define the parameter grid to search
    param_grid = {"alpha": np.logspace(-4, 4, 20)}

    # Setup Cross-Validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Setup Randomized Grid Search
    lasso_cv = RandomizedSearchCV(lasso, param_grid, cv=kfold, random_state=42)

    # Fit the model
    lasso_cv.fit(X_train, y_train)

    # Print the best parameters and score
    print(f"Best parameters found: {lasso_cv.best_params_}")
    print(f"Best cross-validation score: {lasso_cv.best_score_}")

    # Fitting the model
    lasso_cv.fit(X_train, y_train)

    y_pred = lasso_cv.predict(X_test)

    y_pred_train = lasso_cv.predict(X_train)

    mse_test = mean_squared_error(y_test, y_pred)

    mape_test = mean_absolute_percentage_error(y_test, y_pred)

    print("Mean Squared Error on Test Data:", mse_test)

    print("Mean Absolute Percentage Error on Test Data:", mape_test)

    best_lasso = lasso_cv.best_estimator_

    coefficients = best_lasso.coef_

    feature_names = X_train.columns

    # Filter the coefficients and corresponding feature names
    non_zero_coefficients = coefficients[coefficients != 0]
    non_zero_features = feature_names[coefficients != 0]

    # Print the non-zero coefficients and their corresponding feature names
    for feature, coef in zip(non_zero_features, non_zero_coefficients):
        print(f"{feature}: {coef}")

    return {
        "best_lasso": best_lasso,
        "y_pred": y_pred,
        "mse_test": mse_test,
        "mape_test": mape_test,
        "y_test": y_test,
        "X_test_dates": X_test_dates,
        "X_train": X_train,
        "X_train_dates": X_train_dates,
        "y_train": y_train,
        "y_pred_train": y_pred_train,
    
    }
