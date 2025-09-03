from src.logger.logger import get_logger
from src.exception.custom_exception import CustomException
import sys

logger = get_logger(__name__)


def div_numb(a, b):
    try:
        res = a / b
        logger.info("dividing two numbers")
        return res
    except Exception as e:
        logger.error("Error occured")
        raise CustomException("Custom Error Zero", sys)


if __name__ == "__main__":
    try:
        logger.info("Starting main program")
        div_numb(10, 0)
    except CustomException as ce:
        logger.error(str(ce))
