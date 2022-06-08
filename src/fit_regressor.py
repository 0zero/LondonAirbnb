from typing import List
import numpy as np
from joblib import dump
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import (
    LinearRegression,
    RANSACRegressor,
    HuberRegressor,
    Ridge,
    Lasso,
    ElasticNet,
)
from sklearn.ensemble import RandomForestRegressor

# TODO: Turn this into a useful class
# TODO: add missing type information


def build_linear_model(
    numeric_cols: List[str],
    categorical_cols: List[str],
    model_name: str = "ols",
):
    # TODO: add return type
    """
    Build linear model pipeline
    :param numeric_cols: column names representing numeric features
    :param categorical_cols: column names representing categorical features
    :param model_name: name of linear model to use
    :return: complete pipeline object
    """
    if model_name == "ols":
        model = LinearRegression()
    elif model_name == "ransac":
        model = RANSACRegressor()
    elif model_name == "huber":
        model = HuberRegressor()
    elif model_name == "ridge":
        model = Ridge()
    elif model_name == "lasso":
        model = Lasso()
    elif model_name == "elastic":
        model = ElasticNet()
    else:
        msg = (
            "Have you entered the correct Linear Model Regressor? Valid values are: "
            "ols, ransac, huber, ridge, lasso, and elastic."
        )
        raise ValueError(msg)

    numerical_pipeline = Pipeline(
        [
            ("std_scaler", StandardScaler()),
        ]
    )
    data_pipeline = ColumnTransformer(
        [
            ("num", numerical_pipeline, numeric_cols),
            ("cat", OneHotEncoder(), categorical_cols),
        ]
    )
    full_pipeline = Pipeline(
        [
            ("data", data_pipeline),
            ("regression", model),
        ]
    )
    # TODO: Use GridSearchCV to find best parameters for models
    return full_pipeline


def build_randforest_model(numeric_cols: List[str], categorical_cols: List[str]):
    """
    Build Random Forest regressor model pipeline
    :param numeric_cols: column names representing numeric features
    :param categorical_cols: column names representing categorical features
    :return: complete pipeline object
    """
    numerical_pipeline = Pipeline(
        [
            ("std_scaler", StandardScaler()),
        ]
    )
    data_pipeline = ColumnTransformer(
        [
            ("num", numerical_pipeline, numeric_cols),
            ("cat", OneHotEncoder(), categorical_cols),
        ]
    )
    full_pipeline = Pipeline(
        [
            ("data", data_pipeline),
            ("regression", RandomForestRegressor()),
        ]
    )

    # TODO: Use GridSearchCV to find best parameters for models
    return full_pipeline


def evaluate_model(model, x_test, y_test):
    """
    Evaluate a given model and return output metrics
    :param model: model object to evaluate
    :param x_test: test input data
    :param y_test: test expected output data
    :return: r2 and root mean squared error metrics for given model
    """
    y_prediction = model.predict(x_test)
    return r2_score(y_test, y_prediction), np.sqrt(
        mean_squared_error(y_test, y_prediction)
    )


def save_model(model, model_filepath):
    """
    Export trained model
    :param model: model object to save
    :param model_filepath: path to model save file
    :return:
    """
    # TODO: use this function
    dump(model, model_filepath)
