import os
import sys
import numpy as np
import pandas as pd
import dill 
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params=None):
    try:
        report = {}

        for model_name, model in models.items():
            # Get parameters for current model
            param_grid = params.get(model_name, {}) if params else {}
            
            if param_grid:
                # Perform GridSearchCV for hyperparameter tuning
                grid_search = GridSearchCV(
                    estimator=model, 
                    param_grid=param_grid, 
                    cv=3, 
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                # Get the best model
                model = grid_search.best_estimator_
            else:
                # Train model without hyperparameter tuning
                model.fit(X_train, y_train)

            # Predict Training data
            y_train_pred = model.predict(X_train)

            # Predict Testing data
            y_test_pred = model.predict(X_test)

            # Get R2 scores for train and test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)