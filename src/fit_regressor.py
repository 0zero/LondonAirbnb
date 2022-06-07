from typing import List
import numpy as np
from joblib import dump
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor

# TODO: Turn this into a useful class


def build_linear_model(
        numeric_cols: List[str],
        categorical_cols: List[str],
        model_name: str = "ols",
):
    """

    :param numeric_cols:
    :param categorical_cols:
    :param model_name:
    :return:
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

    numerical_pipeline = Pipeline([
        ("std_scaler", StandardScaler()),
    ])
    data_pipeline = ColumnTransformer([
        ("num", numerical_pipeline, numeric_cols),
        ("cat", OneHotEncoder(), categorical_cols),
    ])
    full_pipeline = Pipeline([
        ("data", data_pipeline),
        ("regression", model),
    ])
    # TODO: Use GridSearchCV to find best parameters for models
    return full_pipeline


def build_randforest_model(numeric_cols: List[str], categorical_cols: List[str]):
    """

    :param numeric_cols:
    :param categorical_cols:
    :return:
    """
    numerical_pipeline = Pipeline([
        ("std_scaler", StandardScaler()),
    ])
    data_pipeline = ColumnTransformer([
        ("num", numerical_pipeline, numeric_cols),
        ("cat", OneHotEncoder(), categorical_cols),
    ])
    full_pipeline = Pipeline([
        ("data", data_pipeline),
        ("regression", RandomForestRegressor()),
    ])

    # TODO: Use GridSearchCV to find best parameters for models
    return full_pipeline


def evaluate_model(model, x_test, y_test):
    """

    :param model:
    :param x_test:
    :param y_test:
    :return:
    """
    y_prediction = model.predict(x_test)
    return r2_score(y_test, y_prediction), np.sqrt(mean_squared_error(y_test, y_prediction))


def save_model(model, model_filepath):
    """

    :param model:
    :param model_filepath:
    :return:
    """
    dump(model, model_filepath)



