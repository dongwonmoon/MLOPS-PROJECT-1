from src.services.data_ingestion import DataIngestion
from src.services.data_preprocessing import DataProcessor
from src.services.model_training import ModelTraining
from utils.common_fuctions import read_yaml
from config.path_config import *

if __name__ == "__main__":
    data_ingestion = DataIngestion(read_yaml(CONFIG_PATH))
    data_ingestion.run()

    processor = DataProcessor(
        TRAIN_FILE_PATH, TEST_FILE_PATH, PROCESSED_DIR, CONFIG_PATH
    )
    processor.process()

    trainer = ModelTraining(
        PROCESSED_TRAIN_DATA_PATH, PROCESSED_TEST_DATA_PATH, MODEL_OUTPUT_PATH
    )
    trainer.run()
