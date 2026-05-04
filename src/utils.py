import os
import sys

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluate models using RandomizedSearchCV.

    IMPORTANT: X_train passed here must be RAW (no SMOTE pre-applied).
    SMOTE now lives inside the ImbPipeline so it only fires within each
    training fold — validation folds never see synthetic samples.
    """
    try:
        report = {}

        for model_name, model in models.items():
            para = param[model_name]

           
            pipe_para = {f"clf__{k}": v for k, v in para.items()}

           
            pipe = ImbPipeline([
                ("smote", SMOTETomek(random_state=42)),
                ("clf",   model)
            ])

           
            gs = RandomizedSearchCV(
                pipe, pipe_para,
                scoring="f1_macro",   
                cv=5,
                n_iter=20,           
                n_jobs=-1,
                random_state=42,
                refit=True
            )
            gs.fit(X_train, y_train)

            y_train_pred = gs.predict(X_train)
            y_test_pred  = gs.predict(X_test)

            train_macro_f1 = f1_score(y_train, y_train_pred, average="macro", zero_division=0)
            test_macro_f1  = f1_score(y_test,  y_test_pred,  average="macro", zero_division=0)

            train_recall   = recall_score(y_train, y_train_pred, zero_division=0)
            test_recall    = recall_score(y_test,  y_test_pred,  zero_division=0)

            
            report[model_name] = test_macro_f1

            print(f"\n{'='*50}")
            print(f"  Model          : {model_name}")
            print(f"  Best CV Params : {gs.best_params_}")
            print(f"  Train Macro F1 : {train_macro_f1:.4f} | Test Macro F1 : {test_macro_f1:.4f}")
            print(f"  Train Recall   : {train_recall:.4f}   | Test Recall   : {test_recall:.4f}")
            if train_macro_f1 - test_macro_f1 > 0.10:
                print(f"  ⚠️  Possible overfit — gap: {train_macro_f1 - test_macro_f1:.4f}")
            print(f"{'='*50}")

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)