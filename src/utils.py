import os
import sys

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import RandomizedSearchCV

# BUG 2 FIX: Import ImbPipeline + SMOTETomek so SMOTE runs INSIDE each CV fold
# Previously SMOTE was applied in data_transformation.py before CV, causing
# synthetic samples to leak into validation folds → inflated CV scores (gap of 0.15)
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

            # ── BUG 3 FIX: Prefix all param keys with "clf__" ─────────────────
            # When using ImbPipeline, GridSearch/RandomizedSearch sees the pipeline
            # as the estimator. Param names must match the step name: "clf__<param>".
            # Without this, RandomizedSearchCV silently ignores all hyperparameters
            # and only evaluates the default model config.
            pipe_para = {f"clf__{k}": v for k, v in para.items()}

            # ── BUG 2 FIX: SMOTE inside pipeline ──────────────────────────────
            # SMOTETomek = SMOTE + Tomek link removal. Cleaner decision boundaries
            # than plain SMOTE, especially at Germany/non-Germany cluster boundaries.
            pipe = ImbPipeline([
                ("smote", SMOTETomek(random_state=42)),
                ("clf",   model)
            ])

            # ── BUG 1 FIX: scoring='f1_macro' (was 'recall') ─────────────────
            # model_trainer.py selects the best model using model_report values
            # and logs them as "Macro F1". But this function was using scoring='recall'
            # for CV tuning and returning test_recall — a direct contradiction.
            # Consistent scoring across tuning, selection, and logging.
            gs = RandomizedSearchCV(
                pipe, pipe_para,
                scoring="f1_macro",   # was: scoring='recall' — INCONSISTENT
                cv=5,
                n_iter=20,            # reduced from 30 since pipe_para covers the space
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

            # ── BUG 1 FIX: return test_macro_f1 (was test_recall) ─────────────
            # model_trainer.py does: best_model_score = max(model_report.values())
            # and logs it as "Macro F1". Returning test_recall here made that log
            # completely wrong — the "best model" was actually best by recall,
            # not by Macro F1 as advertised.
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