import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_dataset(data_path):
    data = pd.read_csv(data_path)
    return data

def encode(data):
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ohe = OneHotEncoder(sparse=False)
    neighborhood_encoded = ohe.fit_transform(data[["Neighborhood"]])
    neighborhood_encoded_cols = [f"Neighborhood_{category}" for category in ohe.categories_[0]]
    neighborhood_df = pd.DataFrame(neighborhood_encoded, columns=neighborhood_encoded_cols)
    df_encoded = pd.concat([data.drop("Neighborhood", axis=1), neighborhood_df], axis=1)
    x_train = df_encoded.drop("Price", axis=1)
    y_train = df_encoded["Price"]
    return x_train, x_test, y_train, y_test
    
