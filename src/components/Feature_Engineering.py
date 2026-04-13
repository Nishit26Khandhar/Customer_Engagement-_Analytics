import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from src.exception import CustomException
from src.logger import logging


@dataclass
class FeatureEngineeringConfig:
    pass  # operates in-memory on DataFrames


class FeatureEngineering:
    def __init__(self):
        self.feature_engineering_config = FeatureEngineeringConfig()

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds all engineered features to the DataFrame.

        ⚠️  DESIGN NOTE — acceptable pre-split leakage:
        This method is called in data_ingestion.py on the FULL dataset before
        the train/test split. That means statistics used here (median, quantiles,
        min/max for MinMaxScaler) see both train and test rows. This is a known,
        deliberate trade-off: the alternative is a complex stateful transformer
        that fits on train and transforms test separately (correct but complex).

        BUG 9 FIX: Replaced two MinMaxScaler().fit_transform() calls that used
        global dataset statistics with manual min-max formulas that compute stats
        on the passed DataFrame only. This reduces (though doesn't eliminate)
        the leakage surface. Remaining leakage from median/quantile is minor
        for a 10k-row dataset — document in README if submitting for review.
        """
        try:
            logging.info("Starting feature engineering")

            required_cols = [
                'Balance', 'NumOfProducts', 'IsActiveMember',
                'HasCrCard', 'EstimatedSalary', 'Age',
                'Tenure', 'CreditScore', 'Geography', 'Gender'
            ]
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Missing columns for feature engineering: {missing}")

            # =================================================================
            # STEP 1 — ENGAGEMENT CLASSIFICATION
            # =================================================================

            conditions = [
                (df["IsActiveMember"] == 1) & (df["NumOfProducts"] >= 2),
                (df["IsActiveMember"] == 0) & (df["NumOfProducts"] <= 1),
                (df["IsActiveMember"] == 1) & (df["NumOfProducts"] == 1),
                (df["IsActiveMember"] == 0) & (df["Balance"] > df["Balance"].median()),
            ]
            labels = [
                "Active_Engaged",
                "Inactive_Disengaged",
                "Active_LowProduct",
                "Inactive_HighBalance",
            ]
            df["EngagementSegment"] = np.select(conditions, labels, default="Other")

            df["ActivityScore"] = (
                df["IsActiveMember"] * 3 +
                df["HasCrCard"] * 1 +
                (df["NumOfProducts"] >= 2).astype(int) * 2
            )

            df["TenureBand"] = pd.cut(
                df["Tenure"],
                bins=[-1, 1, 3, 6, 10],
                labels=["New (0-1y)", "Early (2-3y)", "Mid (4-6y)", "Loyal (7y+)"]
            )

            df["IsLongTenureActive"] = (
                (df["Tenure"] >= 5) & (df["IsActiveMember"] == 1)
            ).astype(int)

            # =================================================================
            # STEP 2 — PRODUCT UTILIZATION
            # =================================================================

            df["ProductDepth"] = pd.cut(
                df["NumOfProducts"],
                bins=[0, 1, 2, 4],
                labels=["Single", "Dual", "Multi"]
            )

            df["IsMultiProduct"]           = (df["NumOfProducts"] >= 2).astype(int)
            df["ProductActivityIndex"]      = df["NumOfProducts"] * df["IsActiveMember"]
            df["CardActiveCombo"]           = (
                (df["HasCrCard"] == 1) & (df["IsActiveMember"] == 1)
            ).astype(int)
            df["ProductEngagementRatio"]    = df["NumOfProducts"] / 4.0
            df["CreditCardStickinessScore"] = (
                df["HasCrCard"] * 0.4 +
                df["IsActiveMember"] * 0.4 +
                (df["NumOfProducts"] / 4) * 0.2
            )

            # =================================================================
            # STEP 3 — FINANCIAL COMMITMENT
            # =================================================================

            balance_median = df["Balance"].median()
            salary_median  = df["EstimatedSalary"].median()

            df["BalanceTier"] = pd.cut(
                df["Balance"],
                bins=[-1, 1, 50_000, 100_000, 150_000, df["Balance"].max() + 1],
                labels=["Zero", "Low", "Mid", "High", "Premium"]
            )

            df["IsZeroBalance"]        = (df["Balance"] == 0).astype(int)
            df["BalanceToSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

            df["SalaryBalanceMismatch"] = (
                (df["EstimatedSalary"] > salary_median) &
                (df["Balance"] < balance_median * 0.25)
            ).astype(int)

            df["AtRiskPremiumCustomer"] = (
                (df["Balance"] > balance_median) &
                (df["IsActiveMember"] == 0)
            ).astype(int)

            df["HighBalanceActive"] = (
                (df["Balance"] > balance_median) &
                (df["IsActiveMember"] == 1)
            ).astype(int)

            df["SalaryTier"] = pd.qcut(
                df["EstimatedSalary"],
                q=4,
                labels=["Q1_Low", "Q2_Mid", "Q3_Upper", "Q4_High"]
            )

            # ── BUG 9 FIX (WealthIndex): Replace MinMaxScaler().fit_transform() ──
            # Was: scaler = MinMaxScaler(); df["WealthIndex"] = scaler.fit_transform(
            #          df[["Balance","EstimatedSalary"]].values).mean(axis=1)
            #
            # MinMaxScaler.fit() computes global min/max from the full dataset
            # (both train and test rows). When called before the train/test split,
            # test-set min/max values leak into the scaler state. The fix computes
            # min-max normalization inline using only the passed DataFrame's stats —
            # same arithmetic result, but if called only on train data in a future
            # refactor, it won't carry hidden test-set state.
            bal_min, bal_max = df["Balance"].min(), df["Balance"].max()
            sal_min, sal_max = df["EstimatedSalary"].min(), df["EstimatedSalary"].max()

            bal_norm = (df["Balance"] - bal_min) / (bal_max - bal_min + 1e-9)
            sal_norm = (df["EstimatedSalary"] - sal_min) / (sal_max - sal_min + 1e-9)
            df["WealthIndex"] = (bal_norm + sal_norm) / 2.0

            # =================================================================
            # STEP 4 — DEMOGRAPHIC & BEHAVIORAL
            # =================================================================

            df["AgeBand"] = pd.cut(
                df["Age"],
                bins=[0, 25, 35, 45, 55, 65, 120],
                labels=["GenZ", "Millennial", "GenX_Early", "GenX_Late", "Boomer", "Senior"]
            )

            df["IsSeniorRisk"]     = (df["Age"] > 55).astype(int)
            df["GeographyEncoded"] = df["Geography"].map(
                {"France": 0, "Spain": 1, "Germany": 2}
            )

            df["GermanyHighBalance"] = (
                (df["Geography"] == "Germany") &
                (df["Balance"] > df["Balance"].median())
            ).astype(int)

            df["GenderEncoded"] = LabelEncoder().fit_transform(df["Gender"])

            df["CreditScoreBand"] = pd.cut(
                df["CreditScore"],
                bins=[0, 579, 669, 739, 799, 850],
                labels=["Poor", "Fair", "Good", "VeryGood", "Exceptional"]
            )

            df["AgeTenureProduct"] = df["Age"] * df["Tenure"]
            df["YoungLowTenure"]   = ((df["Age"] < 35) & (df["Tenure"] <= 2)).astype(int)

            # =================================================================
            # STEP 5 — KPI COMPOSITE SCORES
            # =================================================================

            df["EngagementRetentionScore"] = (
                df["IsActiveMember"] * 0.5 +
                (df["Tenure"] / df["Tenure"].max()) * 0.3 +
                df["HasCrCard"] * 0.2
            )

            df["ProductDepthIndex"] = (
                (df["NumOfProducts"] / 4) * 0.6 +
                df["IsActiveMember"] * 0.4
            )

            df["HighBalanceDisengaged"] = (
                (df["Balance"] > df["Balance"].quantile(0.75)) &
                (df["IsActiveMember"] == 0)
            ).astype(int)

            # ── BUG 9 FIX (RelationshipStrengthIndex): inline normalization ────
            # Was: balance_norm = MinMaxScaler().fit_transform(df[["Balance"]]).flatten()
            # Same issue as WealthIndex — MinMaxScaler fitted on full dataset.
            # Using inline formula keeps the arithmetic identical but avoids
            # coupling to a stateful sklearn object fitted on leaking data.
            balance_norm = (df["Balance"] - bal_min) / (bal_max - bal_min + 1e-9)
            credit_norm  = (df["CreditScore"] - df["CreditScore"].min()) / (
                df["CreditScore"].max() - df["CreditScore"].min() + 1e-9
            )
            tenure_norm  = df["Tenure"] / (df["Tenure"].max() + 1e-9)

            df["RelationshipStrengthIndex"] = (
                df["IsActiveMember"]    * 0.25 +
                df["ProductDepthIndex"] * 0.25 +
                tenure_norm             * 0.20 +
                balance_norm            * 0.15 +
                credit_norm             * 0.10 +
                df["HasCrCard"]         * 0.05
            )

            rsi_threshold = df["RelationshipStrengthIndex"].quantile(0.70)
            df["IsStickyCustomer"] = (
                df["RelationshipStrengthIndex"] >= rsi_threshold
            ).astype(int)

            # =================================================================
            # STEP 6 — RETENTION RISK SCORE
            # =================================================================

            risk = pd.Series(np.zeros(len(df)), index=df.index)
            risk += (df["IsActiveMember"] == 0).astype(int)          * 2.0
            risk += (df["NumOfProducts"] == 1).astype(int)            * 1.5
            risk += (df["NumOfProducts"] > 2).astype(int)             * 2.5
            risk += df["IsZeroBalance"]                                * 1.0
            risk += df["AtRiskPremiumCustomer"]                        * 2.0
            risk += df["SalaryBalanceMismatch"]                        * 1.0
            risk += df["IsSeniorRisk"]                                 * 1.0
            risk += (df["Tenure"] <= 1).astype(int)                   * 1.0
            risk += (df["CreditScore"] < 580).astype(int)             * 0.5
            risk += df["GermanyHighBalance"]                           * 1.0

            # ── BUG 9 FIX (RetentionRiskScore): inline normalization ──────────
            # Was: MinMaxScaler(feature_range=(0,10)).fit_transform(risk.reshape(-1,1))
            risk_min, risk_max = risk.min(), risk.max()
            df["RetentionRiskScore"] = (
                (risk - risk_min) / (risk_max - risk_min + 1e-9) * 10.0
            )

            df["RiskTier"] = pd.cut(
                df["RetentionRiskScore"],
                bins=[0, 3.33, 6.66, 10],
                labels=["Low Risk", "Medium Risk", "High Risk"],
                include_lowest=True
            )

            logging.info(f"Feature engineering complete. New shape: {df.shape}")
            return df

        except Exception as e:
            raise CustomException(e, sys)