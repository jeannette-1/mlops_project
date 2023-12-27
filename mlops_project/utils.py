import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def load_dataset(data_path):
    data = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def encode(data):
    ohe = OneHotEncoder(sparse=False)
    neighborhood_encoded = ohe.fit_transform(df[["Neighborhood"]])
    neighborhood_encoded_cols = [f"Neighborhood_{category}" for category in ohe.categories_[0]]
    neighborhood_df = pd.DataFrame(neighborhood_encoded, columns=neighborhood_encoded_cols)
    df_encoded = pd.concat([df.drop("Neighborhood", axis=1), neighborhood_df], axis=1)
    X_train = df_encoded.drop("Price", axis=1)
    y_train = df_encoded["Price"]
    return X_train, X_test, y_train, y_test
    
