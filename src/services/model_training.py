import os
import pandas as pd
import joblib
import sys
from scipy.stats import randint
import mlflow
import mlflow.sklearn
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from src.logger.logger import get_logger
from src.exception.custom_exception import CustomException
from config.path_config import *
from config.model_params import *
from utils.common_fuctions import read_yaml, load_data


logger = get_logger(__name__)


class ModelTraining:
    def __init__(self, train_path, test_path, model_output_path):
        self.train_path = train_path
        self.test_path = test_path
        self.model_output_path = model_output_path

        self.params_dist = LIGHTGBM_PARAMS
        self.random_search_params = RANDOM_SEARCH_PARAMS

    def load_and_split_data(self):
        try:
            logger.info(f"Loading Data from {self.train_path}")
            train_df = load_data(self.train_path)

            logger.info(f"Loading Data from {self.test_path}")
            test_df = load_data(self.test_path)

            X_train = train_df.drop(columns=["booking_status"])
            y_train = train_df["booking_status"]

            X_test = test_df.drop(columns=["booking_status"])
            y_test = test_df["booking_status"]

            logger.info("Data splitted successfully for Model Training")

            return X_train, y_train, X_test, y_test

        except Exception as e:
            logger.error("Error while loading and splitting data")
            raise CustomException("FAILED to load and split data", sys)

    def train_lgbm(self, X_train, y_train):
        try:
            logger.info("Initializing our Model")
            model = lgb.LGBMClassifier(
                random_state=self.random_search_params["random_state"]
            )

            logger.info("Starting our Hyperparameter Tuning")
            random_search = RandomizedSearchCV(
                estimator=model,
                param_distributions=self.params_dist,
                n_iter=self.random_search_params["n_iter"],
                cv=self.random_search_params["cv"],
                n_jobs=self.random_search_params["n_jobs"],
                verbose=self.random_search_params["verbose"],
                random_state=self.random_search_params["random_state"],
                scoring=self.random_search_params["scoring"],
            )

            logger.info("Training our Model")
            random_search.fit(X_train, y_train)

            logger.info("Hyperparameter Tuning Completed")

            best_params = random_search.best_params_
            best_model = random_search.best_estimator_

            logger.info(f"Best params: {best_params}")

            return best_model

        except Exception as e:
            logger.error("Error while training the model")
            raise CustomException("FAILED to train the model", sys)

    def evaluate_model(self, model, X_test, y_test):
        try:
            logger.info("Evaluating our Model")

            y_pred = model.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logger.info(f"Accuracy: {accuracy}")
            logger.info(f"Precision: {precision}")
            logger.info(f"Recall: {recall}")
            logger.info(f"F1 Score: {f1}")

            return {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            }

        except Exception as e:
            logger.error("Error while evaluating the model")
            raise CustomException("FAILED to evaluate the model", sys)

    def save_model(self, model):
        try:
            logger.info("Saving our Model")
            os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
            joblib.dump(model, self.model_output_path)

            logger.info(f"Model saved to {self.model_output_path}")

        except Exception as e:
            logger.error("Error while saving the model")
            raise CustomException("FAILED to save the model", sys)

    def run(self):
        try:
            with mlflow.start_run():
                logger.info("Starting our Model Training pipeline")

                logger.info("Starting our MLflow Experimentation")

                logger.info("Logging the training and testing dataset to MLflow")
                mlflow.log_artifact(self.train_path, artifact_path="datasets")
                mlflow.log_artifact(self.test_path, artifact_path="datasets")

                X_train, y_train, X_test, y_test = self.load_and_split_data()

                model = self.train_lgbm(X_train, y_train)

                metrics = self.evaluate_model(model, X_test, y_test)

                self.save_model(model)

                logger.info("Logging the model into MLflow")
                mlflow.log_artifact(self.model_output_path)

                logger.info("Logging Params and Metrics into MLflow")
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)

                logger.info("Model Training pipeline completed successfully")

        except Exception as e:
            logger.error("Error while running the pipeline")
            raise CustomException("FAILED to run the pipeline", sys)


if __name__ == "__main__":
    model_trainer = ModelTraining(
        PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH
    )
    model_trainer.run()
