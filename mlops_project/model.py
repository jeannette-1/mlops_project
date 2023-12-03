import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score


class CatBoostModel:
    def __init__(self, name: str, model: CatBoostRegressor, params: dict = None):
        self.name = name
        self.model = model
        self.params = params

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics = pd.DataFrame({
            "Model Name": [self.name],
            "MSE": [mse],
            "R2 Score": [r2]
        })
        return metrics
