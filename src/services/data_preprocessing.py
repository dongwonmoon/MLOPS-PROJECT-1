import os
import pandas as pd
import numpy as np
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

from src.logger.logger import get_logger
from src.exception.custom_exception import CustomException
from config.path_config import *
from utils.common_fuctions import read_yaml, load_data


logger = get_logger(__name__)


class DataProcessor:
    def __init__(self, train_path, test_path, processed_dir, config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_dir = processed_dir

        self.config = read_yaml(config_path)

        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def preprocess_data(self, df):
        try:
            logger.info("Preprocessing data...")

            logger.info("Dropping columns")
            df.drop(["Booking_ID"], axis=1, inplace=True)

            logger.info("Dropping duplicates")
            df.drop_duplicates(inplace=True)

            cat_cols = self.config["data_processing"]["categorical_columns"]
            num_cols = self.config["data_processing"]["numerical_columns"]

            logger.info("Label Encoding categorical columns")
            le = LabelEncoder()
            mappings = {}

            for col in cat_cols:
                df[col] = le.fit_transform(df[col])
                mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

            logger.info("Label Mappings are:")
            for col, mapping in mappings.items():
                logger.info(f"{col}: {mapping}")

            logger.info("Doing Skewness Handling")
            skew_threshold = self.config["data_processing"]["skewness_threshold"]
            for col in num_cols:
                if df[col].skew() > skew_threshold:
                    df[col] = np.log1p(df[col])

            return df

        except Exception as e:
            logger.error("Error while preprocessing data")
            raise CustomException("FAILED to preprocess data", sys)

    def balance_data(self, df):
        try:
            logger.info("Balancing data...")
            X = df.drop("booking_status", axis=1)
            y = df["booking_status"]

            smote = SMOTE(random_state=42)

            X_res, y_res = smote.fit_resample(X, y)

            df_res = pd.concat([X_res, y_res], axis=1)

            return df_res

        except Exception as e:
            logger.error("Error while balancing data")
            raise CustomException("FAILED to balance data", sys)

    def select_features(self, df):
        try:
            logger.info("Selecting features...")

            X = df.drop("booking_status", axis=1)
            y = df["booking_status"]

            model = RandomForestClassifier(random_state=42)
            model.fit(X, y)

            feature_importances = model.feature_importances_

            feature_importances_df = pd.DataFrame(
                {"Feature": X.columns, "Importance": feature_importances}
            )
            top_features = feature_importances_df.sort_values(
                by="Importance", ascending=False
            )

            top_10_features = top_features.head(10)

            selected_features = top_10_features["Feature"].tolist()

            return df[selected_features + ["booking_status"]]

        except Exception as e:
            logger.error("Error while selecting features")
            raise CustomException("FAILED to select features", sys)

    def save_data(self, df, file_path):
        try:
            logger.info("Saving data...")
            df.to_csv(file_path, index=False)
            logger.info(f"Data saved to {file_path}")

        except Exception as e:
            logger.error("Error while saving data")
            raise CustomException("FAILED to save data", sys)

    def process(self):
        try:
            logger.info("Loading Data From RAW Directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.select_features(train_df)
            test_df = self.select_features(test_df)

            self.save_data(train_df, PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df, PROCESSED_TEST_DATA_PATH)

        except Exception as e:
            logger.error("Error while processing data")
            raise CustomException("FAILED to process data", sys)


if __name__ == "__main__":
    data_processor = DataProcessor(
        TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH
    )
    data_processor.process()
