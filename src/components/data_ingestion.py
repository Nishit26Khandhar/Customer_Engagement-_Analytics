import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.components.Feature_Engineering import FeatureEngineering
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path:   str = os.path.join("artifacts", "raw.csv")
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path:  str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, source_path: str):
        """
        End-to-end pipeline:
          1. Load raw CSV
          2. Validate core columns
          3. Apply feature engineering
          4. Stratified train/test split (stratify on Exited)
          5. Save artifacts
          6. Return paths for DataTransformation
        """
        logging.info("=" * 55)
        logging.info("DATA INGESTION STARTED")
        logging.info("=" * 55)

        try:
            # ── 1. Load ──────────────────────────────────────────────────────
            logging.info(f"Reading dataset from: {source_path}")
            df = pd.read_csv(source_path)
            logging.info(f"Raw dataset shape: {df.shape}")

            # ── 2. Validate ──────────────────────────────────────────────────
            required_raw_cols = [
                "CreditScore", "Geography", "Gender", "Age", "Tenure",
                "Balance", "NumOfProducts", "HasCrCard",
                "IsActiveMember", "EstimatedSalary", "Exited"
            ]
            missing = [c for c in required_raw_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Drop irrelevant identity columns if present
            df.drop(columns=["CustomerId", "Surname", "RowNumber"],
                    inplace=True, errors="ignore")

           
            for col in ["HasCrCard", "IsActiveMember", "Exited"]:
                non_binary = ~df[col].isin([0, 1])
                if non_binary.any():
                    bad_vals = df.loc[non_binary, col].unique().tolist()
                    raise ValueError(
                        f"Non-binary values found in '{col}': {bad_vals}. "
                        f"Expected only 0 or 1."
                    )

            logging.info("Column validation passed")

            # ── 3. Feature Engineering ───────────────────────────────────────
            logging.info("Applying feature engineering...")
            fe = FeatureEngineering()
            df = fe.engineer_features(df)
            logging.info(f"Post-engineering shape: {df.shape}")

            # ── 4. Save raw (post-engineering) artifact ──────────────────────
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False)
            logging.info(f"Raw (engineered) data saved → {self.ingestion_config.raw_data_path}")

            # ── 5. Stratified Train / Test Split ─────────────────────────────
            logging.info("Performing stratified train/test split (80/20)...")
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
                stratify=df["Exited"]
            )

            train_churn = train_set["Exited"].mean()
            test_churn  = test_set["Exited"].mean()
            logging.info(f"Train size: {len(train_set):,} | Churn rate: {train_churn:.2%}")
            logging.info(f"Test size : {len(test_set):,}  | Churn rate: {test_churn:.2%}")

            # ── 6. Save split artifacts ──────────────────────────────────────
            train_set.to_csv(self.ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,   index=False)
            logging.info(f"Train data saved → {self.ingestion_config.train_data_path}")
            logging.info(f"Test data saved  → {self.ingestion_config.test_data_path}")

            logging.info("DATA INGESTION COMPLETE")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion("notebook/data/European_Bank.csv")

    data_transformation = DataTransformation()
    train_array, test_array, preprocessor_path = data_transformation.initiate_data_transformation(
        train_path, test_path
    )

    model_trainer = ModelTrainer()
    report = model_trainer.initiate_model_trainer(train_array, test_array)
    print("\nClassification Report:\n", report)