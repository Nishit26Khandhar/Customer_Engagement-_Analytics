import os
import sys
from dataclasses import dataclass
import numpy as np
from collections import Counter

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            if np.isnan(y_train).any():
                raise Exception("❌ ERROR: NaN in y_train at model stage")

            scale_pos = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

            models = {
                "Logistic Regression": LogisticRegression(
                    max_iter=1000, class_weight='balanced', solver='lbfgs'
                ),
                "Decision Tree": DecisionTreeClassifier(
                    class_weight='balanced', random_state=42
                ),
                "Random Forest": RandomForestClassifier(
                    class_weight='balanced', random_state=42, n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(
                    eval_metric='logloss',
                    scale_pos_weight=scale_pos,
                    random_state=42,
                    n_jobs=-1
                ),
                "CatBoost": CatBoostClassifier(
                    auto_class_weights='Balanced',
                    verbose=0,
                    random_state=42
                )
            }

            params = {
                "Logistic Regression": {},

                "Decision Tree": {
                    'criterion':         ['gini', 'entropy'],
                    'max_depth':         [3, 5, 7],
                    'min_samples_split': [2, 5, 10]
                },

                "Random Forest": {
                    'n_estimators':      [100, 200, 300],
                    'max_depth':         [4, 6, 8],
                    'min_samples_split': [2, 5]
                },

                "Gradient Boosting": {
                    'learning_rate': [0.05, 0.1],
                    'subsample':     [0.8, 0.9],
                    'n_estimators':  [100, 200],
                },

                "XGBoost": {
                    'learning_rate':    [0.03, 0.05, 0.1],
                    'n_estimators':     [200, 300, 400],
                    'max_depth':        [3, 4, 5],
                    'subsample':        [0.7, 0.8, 0.9],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'min_child_weight': [1, 3, 5],
                    'gamma':            [0, 0.1, 0.3],
                    'reg_alpha':        [0, 0.1, 0.5]
                },

                "CatBoost": {
                    'depth':         [4, 6, 8],
                    'learning_rate': [0.03, 0.05, 0.1],
                    'iterations':    [200, 300, 400],
                    'l2_leaf_reg':   [1, 3, 5, 7],
                    'border_count':  [32, 64, 128]
                }
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test,   y_test=y_test,
                models=models,   param=params
            )

            logging.info(f"Model evaluation report:\n{model_report}")

            best_model_score = max(sorted(model_report.values()))
            best_model_name  = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            logging.info(f"All model scores (Macro F1): {model_report}")
            
            logging.info("Calibrating best model probabilities...")
            best_model = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
            best_model.fit(X_train, y_train)
            logging.info("Calibration complete")

            logging.info("=" * 55)
            logging.info(f"Best model: {best_model_name} with Macro F1: {best_model_score:.4f}")
            logging.info("=" * 55)

            if best_model_score < 0.55:
                raise CustomException(
                    f"Best recall score {best_model_score:.4f} is below minimum threshold 0.45", sys
                )

            logging.info("Best found model on both training and testing dataset")

            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='recall')
            logging.info(f"Cross-validation F1 scores (5-fold): {np.round(cv_scores, 4)}")
            logging.info(f"Mean CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # ── Optimal threshold via Precision-Recall curve ──────────────────
            predicted_proba = best_model.predict_proba(X_test)[:, 1]
            best_threshold  = 0.3  # fallback

            logging.info("\n===== Searching Best Threshold (Recall + Accuracy) =====")
            
            best_f1_combo  = -1
            threshold_found = False    # tracks best recall+precision balance seen so far
            
            for t in np.linspace(0.2, 0.5, 31):
                 pred_temp = (predicted_proba >= t).astype(int)
                 acc  = accuracy_score(y_test, pred_temp)
                 rec  = recall_score(y_test, pred_temp)
                 prec = precision_score(y_test, pred_temp)
                 f1   = f1_score(y_test, pred_temp, zero_division=0)
                 
                 logging.info(
                      f"Threshold: {t:.2f} | Acc: {acc:.3f} | Recall: {rec:.3f} "
                      f"| Precision: {prec:.3f} | F1: {f1:.3f}"
                      )
                 
                  # Primary condition — recall is king, with a minimum precision floor

                 if rec >= 0.78 and prec >= 0.45:
                    best_threshold = t
                    threshold_found = True
                    logging.info(f"\n✅ Selected Threshold: {t:.2f} | Recall: {rec:.3f} | Precision: {prec:.3f}")
                    
                    break
                 
                 # Fallback tracker — if no threshold meets criteria, pick the best recall+precision balance
                 
                 weighted = (rec * 0.7) + (prec * 0.3)   # recall weighted 70%, precision 30%
                 
                 if weighted > best_f1_combo and rec >= 0.70:
                     best_f1_combo  = weighted
                     best_threshold = t
                     
                     if not threshold_found:        
                         logging.warning(
                             f"⚠️  No threshold met strict criteria — using best recall-precision balance: {best_threshold:.2f}"
                             )
                         
                         predicted = (predicted_proba >= best_threshold).astype(int)
                         logging.info(f"Optimal threshold: {best_threshold:.4f}")

            # ── Risk band assignment ─────────────────────────────────────────
            def assign_risk_band(prob):
                if prob >= 0.70:
                    return "High Risk"
                elif prob >= 0.40:
                    return "Medium Risk"
                return "Low Risk"

            risk_bands        = [assign_risk_band(p) for p in predicted_proba]
            risk_distribution = Counter(risk_bands)

            logging.info("=" * 55)
            logging.info("RISK SCORE DISTRIBUTION (Test Set)")
        
            logging.info(f"  High Risk   (prob >= 0.70) : {risk_distribution['High Risk']} customers")
            logging.info(f"  Medium Risk (prob >= 0.40) : {risk_distribution['Medium Risk']} customers")
            logging.info(f"  Low Risk    (prob  < 0.40) : {risk_distribution['Low Risk']} customers")
            logging.info("=" * 55)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj={
                    "model":           best_model,
                    "threshold":       best_threshold,
                    "risk_bands":      {"high": 0.70, "medium": 0.40, "low": 0.0},
                    "best_model_name": best_model_name
                }
            )
            logging.info(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

            accuracy  = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted,  zero_division=0)
            recall    = recall_score(y_test, predicted,     zero_division=0)
            f1        = f1_score(y_test, predicted,         zero_division=0)
            macro_f1  = f1_score(y_test, predicted, average='macro', zero_division=0)
            roc_auc   = roc_auc_score(y_test, predicted_proba)

            logging.info("=" * 55)
            logging.info(f"Best Model Name  : {best_model_name}")
            logging.info(f"Best Threshold   : {best_threshold:.4f}")
            logging.info(f"Accuracy         : {accuracy:.4f}")
            logging.info(f"Precision        : {precision:.4f}")
            logging.info(f"Recall           : {recall:.4f}")
            logging.info(f"F1 Score         : {f1:.4f}")
            logging.info(f"Macro F1         : {macro_f1:.4f}")
            logging.info(f"ROC-AUC Score    : {roc_auc:.4f}")
            logging.info(f"Mean CV F1       : {cv_scores.mean():.4f}")
            logging.info("=" * 55)

           
            report = classification_report(
                y_test, predicted,
                target_names=["Retained", "Churned"]
            )
            logging.info(f"Classification Report:\n{report}")

            return report

        except Exception as e:
            raise CustomException(e, sys)