import pickle
import sys
from pathlib import Path

import hydra
import mlflow
import pandas as pd

from mlops_project.mlops_project import CatBoostModel
from mlops_project.mlops_project import load_dataset

hydra.main(config_path="../configs", config_name="config", version_base=False)

def infer(config):
    model_name = config["model"]["name"]
    mlflow_models_path = config["common"]["mlflow_models_path"]
    inference_model = config["model"].get("inference_model")
    
    x_train, x_test, y_train, y_test = load_dataset(config["common"]["processed_data_path"])

        model = CatBoostModel(
            name=model_name,
            model=mlflow.catboost.load_model(inference_model or f"{mlflow_models_path}{model_name}/"),
        )

    
    y_pred, metrics = model.evaluate(x_test, y_test)

    print(pd.DataFrame(metrics))


if __name__ == "__main__":
    infer()


