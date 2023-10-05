#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def train_model(X_train, y_train):
    pipeline = Pipeline([("scaler", StandardScaler()),
                       ("basic_logreg", LogisticRegression(multi_class='multinomial',solver='saga',tol=1e-3,max_iter=500))])
    param_grid = {
        'basic_logreg__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'basic_logreg__penalty': ['l1', 'l2'],
    }
    grid_pipeline = GridSearchCV(pipeline,param_grid)
    grid_pipeline.fit(X_train,y_train)
    joblib.dump(grid_pipeline.best_estimator_, 'model.joblib')

