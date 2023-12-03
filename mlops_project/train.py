import sys
from pathlib import Path
import git

import hydra
import mlflow
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from mlops_tools.models import CatBoostModel
from mlops_tools.utils import load_dataset

hydra.main(config_path="../configs", config_name="config", strict=False)

def main(config):
    model_name = config["model"]["name"]
    model_params = config["model"]["params"]
    mlflow_models_path = config["common"]["mlflow_models_path"]
    
    repo = git.Repo(search_parent_directories=True)
    repo_url = repo.remotes[0].config_reader.get("url")
    
    reg = CatBoostModel(model_name, model_params)
    x_train, x_test, y_train, y_test = load_dataset(config["common"]["data_path"])
   
    mlflow.set_tracking_uri(config["common"]["mlflow_uri"])
    mlflow.set_experiment(f"{model_name}_training")

    with mlflow.start_run():
        reg.fit(x_train, y_train)
        mlflow.log_params(reg.model.get_params())
        mlflow.log_param("git commit id", repo.head.commit.hexsha)

        evals_result = pd.DataFrame(reg.model.get_evals_result()["learn"])
        for i, row in evals_result.iterrows():
            mlflow.log_metrics(row, step=i)

        mlflow.catboost.save_model(
            reg.model,
            f"{mlflow_models_path}/{model_name}/",
            signature=mlflow.models.infer_signature(x_train, y_train),
        )

    with open(f"{model_name}.sav", "wb") as f:
        pickle.dump(reg, f)

if __name__ == "__main__":
    main()
