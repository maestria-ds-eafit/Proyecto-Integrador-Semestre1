from sklearn.metrics import mean_squared_error
from lags import create_df_with_lags
import statsmodels.api as sm
from utils import get_significant_variables


def perform_mlr(data):
    df = data["df"]
    if data.get("lags", 0) > 0:
        df = create_df_with_lags(df, data["lags"])
    fecha_corte = data.get("fecha_corte", "2023-07-01")
    X_train = df[df["Date"] < fecha_corte].drop(["Date", "energy_price"], axis=1)
    X_test = df[df["Date"] >= fecha_corte].drop(["Date", "energy_price"], axis=1)
    y_train = df[df["Date"] < fecha_corte]["energy_price"]
    y_test = df[df["Date"] >= fecha_corte]["energy_price"]
    X = sm.add_constant(X_train)

    # Create a model
    model = sm.OLS(y_train, X)

    # Fit the model
    result = model.fit()
    result_summary = result.summary()

    # Print out the statistics
    print(result_summary)

    get_significant_variables(result_summary)

    X_new_with_intercept = sm.add_constant(X_test)
    y_pred = result.predict(X_new_with_intercept)

    # Calculate MSE
    mse = mean_squared_error(y_test, y_pred)

    print(f"Mean Squared Error on Test Data: {mse}")
