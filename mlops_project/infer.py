import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def load_model():
    return joblib.load('model.joblib')

def predict(model, X_test):
    return model.predict(X_test)

def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=None)
    return accuracy, f1

def save_predictions(y_pred):
    df = pd.DataFrame({'prediction': y_pred})
    df.to_csv('predictions.csv', index=False)

