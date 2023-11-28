from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd


def forward_selection(X_train, y_train):
    sfs = SequentialFeatureSelector(LinearRegression(), direction="forward", cv=None)
    sfs.fit(X_train, y_train)
    selected_features = list(X_train.columns[sfs.get_support()])
    if "Date" not in selected_features:
        selected_features.append("Date")
    if "energy_price" not in selected_features:
        selected_features.append("energy_price")

    return selected_features


# Backward Selection
def backward_selection(X_train, y_train):
    sbs = SequentialFeatureSelector(LinearRegression(), direction="backward", cv=None)
    sbs.fit(X_train, y_train)
    selected_features = list(X_train.columns[sbs.get_support()])

    if "Date" not in selected_features:
        selected_features.append("Date")
    if "energy_price" not in selected_features:
        selected_features.append("energy_price")

    return selected_features


# Aplica forward selection y backward selection a tus datos
