import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.preprocessing import StandardScaler
from collections import Counter

# BUG 4 FIX: Removed SMOTE import from here.
# Previously SMOTE was applied here before returning train_array to model_trainer.
# model_trainer then passed that pre-SMOTEd array into evaluate_models → RandomizedSearchCV.
# RandomizedSearchCV split that SMOTE-balanced array into train/validation folds,
# meaning EACH validation fold also contained synthetic samples that were generated
# FROM the training fold data. This is data leakage — CV scores were artificially
# inflated by ~0.10–0.15 (observed: CV F1=0.78 vs real test Macro F1=0.63).
#
# SMOTE now lives inside ImbPipeline in utils.py/evaluate_models(), so it only
# fires on the training portion of each fold. Validation folds stay clean.

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    # ── Ordinal encoding maps (must match Feature_Engineering.py exactly) ─────
    ORDINAL_MAPS = {
        "TenureBand": {
            "New (0-1y)":   0,
            "Early (2-3y)": 1,
            "Mid (4-6y)":   2,
            "Loyal (7y+)":  3,
        },
        "ProductDepth": {
            "Single": 0,
            "Dual":   1,
            "Multi":  2,
        },
        "BalanceTier": {
            "Zero":    0,
            "Low":     1,
            "Mid":     2,
            "High":    3,
            "Premium": 4,
        },
        "AgeBand": {
            "GenZ":       0,
            "Millennial": 1,
            "GenX_Early": 2,
            "GenX_Late":  3,
            "Boomer":     4,
            "Senior":     5,
        },
        "CreditScoreBand": {
            "Poor":        0,
            "Fair":        1,
            "Good":        2,
            "VeryGood":    3,
            "Exceptional": 4,
        },
        "RiskTier": {
            "Low Risk":    0,
            "Medium Risk": 1,
            "High Risk":   2,
        },
        "SalaryTier": {
            "Q1_Low":   0,
            "Q2_Mid":   1,
            "Q3_Upper": 2,
            "Q4_High":  3,
        },
    }

    NOMINAL_COLS = ["Geography", "EngagementSegment", "Gender"]

    NUMERIC_COLS = [
        "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
        "EstimatedSalary", "ActivityScore", "ProductActivityIndex",
        "ProductEngagementRatio", "CreditCardStickinessScore",
        "BalanceToSalaryRatio", "WealthIndex", "AgeTenureProduct",
        "EngagementRetentionScore", "ProductDepthIndex",
        "RelationshipStrengthIndex", "RetentionRiskScore",
        "GeographyEncoded", "GenderEncoded",
    ]

    BINARY_COLS = [
        "HasCrCard", "IsActiveMember",
        "IsMultiProduct", "CardActiveCombo", "IsZeroBalance",
        "SalaryBalanceMismatch", "AtRiskPremiumCustomer", "HighBalanceActive",
        "IsSeniorRisk", "YoungLowTenure", "IsLongTenureActive",
        "GermanyHighBalance", "HighBalanceDisengaged", "IsStickyCustomer",
    ]

    TARGET_COL = "Exited"

    def _apply_ordinal_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, mapping in self.ORDINAL_MAPS.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(mapping)
                if df[col].isna().any():
                    df[col].fillna(df[col].mode()[0], inplace=True)
                df[col] = df[col].astype(int)
                logging.info(f"  Ordinal encoded: {col}")
        return df

    def _apply_ohe(self, df: pd.DataFrame) -> pd.DataFrame:
        nominal_present = [c for c in self.NOMINAL_COLS if c in df.columns]
        if nominal_present:
            df = pd.get_dummies(df, columns=nominal_present, drop_first=True, dtype=int)
            logging.info(f"  One-hot encoded: {nominal_present}")
        return df

    def _drop_remaining_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        leftover = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if leftover:
            df.drop(columns=leftover, inplace=True)
            logging.info(f"  Dropped leftover categoricals: {leftover}")
        return df

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Full transformation pipeline:
          1. Load train/test CSVs
          2. Ordinal encoding
          3. One-hot encoding (nominal)
          4. Drop leftover categoricals
          5. StandardScaler on numeric cols
          6. Return RAW np.array train & test (no SMOTE — handled in utils.py pipeline)
        """
        try:
            logging.info("=" * 55)
            logging.info("DATA TRANSFORMATION STARTED")
            logging.info("=" * 55)

            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

            # ── Ordinal encoding ─────────────────────────────────────────────
            logging.info("Applying ordinal encoding...")
            train_df = self._apply_ordinal_encoding(train_df)
            test_df  = self._apply_ordinal_encoding(test_df)

            # ── One-hot encoding ─────────────────────────────────────────────
            logging.info("Applying one-hot encoding...")
            train_df = self._apply_ohe(train_df)
            test_df  = self._apply_ohe(test_df)

            # ── Align columns (OHE may create different dummies) ─────────────
            train_df, test_df = train_df.align(test_df, join="left", axis=1, fill_value=0)

            # ── Drop remaining categoricals ──────────────────────────────────
            train_df = self._drop_remaining_categoricals(train_df)
            test_df  = self._drop_remaining_categoricals(test_df)

            # ── Null check ───────────────────────────────────────────────────
            for name, frame in [("train", train_df), ("test", test_df)]:
                nulls = frame.isnull().sum().sum()
                if nulls > 0:
                    logging.warning(f"{name} has {nulls} nulls after encoding — filling with median")
                    frame.fillna(frame.median(numeric_only=True), inplace=True)

            # ── Separate features and target ─────────────────────────────────
            input_feature_train = train_df.drop(columns=[self.TARGET_COL])
            input_feature_test  = test_df.drop(columns=[self.TARGET_COL])
            target_train        = train_df[self.TARGET_COL]
            target_test         = test_df[self.TARGET_COL]

            logging.info(f"Features after encoding: {input_feature_train.shape[1]}")
            logging.info(f"Class distribution (raw train): {Counter(target_train.tolist())}")

            # ── StandardScaler on numeric columns ────────────────────────────
            numeric_present = [c for c in self.NUMERIC_COLS if c in input_feature_train.columns]
            logging.info(f"Scaling {len(numeric_present)} numeric columns with StandardScaler")

            scaler = StandardScaler()
            input_feature_train[numeric_present] = scaler.fit_transform(
                input_feature_train[numeric_present]
            )
            input_feature_test[numeric_present] = scaler.transform(
                input_feature_test[numeric_present]
            )

            # ── BUG 4 FIX: No SMOTE here ─────────────────────────────────────
            # train_array is returned RAW (imbalanced). SMOTE is now applied
            # inside ImbPipeline in evaluate_models() (utils.py) so that each
            # CV fold generates its own synthetic samples only from training
            # data, never contaminating validation folds.
            train_array = np.c_[
                input_feature_train.values,
                np.array(target_train)
            ]
            test_array = np.c_[
                input_feature_test.values,
                np.array(target_test)
            ]

            logging.info(f"Train array shape (raw, imbalanced): {train_array.shape}")
            logging.info(f"Test array shape                    : {test_array.shape}")

            # ── Save preprocessor (scaler + column order) ────────────────────
            os.makedirs(
                os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path),
                exist_ok=True
            )
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj={
                    "scaler":          scaler,
                    "numeric_cols":    numeric_present,
                    "feature_columns": list(input_feature_train.columns),
                    "ordinal_maps":    self.ORDINAL_MAPS,
                    "nominal_cols":    self.NOMINAL_COLS,
                }
            )
            logging.info(
                f"Preprocessor saved → {self.data_transformation_config.preprocessor_obj_file_path}"
            )
            logging.info("DATA TRANSFORMATION COMPLETE")

            return train_array, test_array, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)