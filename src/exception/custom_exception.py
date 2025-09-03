import traceback
import sys


class CustomException(Exception):
    def __init__(self, message, detail: sys):
        super().__init__(message)
        self.message = self.get_detail_message(message, detail)

    @staticmethod
    def get_detail_message(message, detail: sys):
        _, _, exc_tb = detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return f"Error in {file_name}, line {line_number}: {message}"

    def __str__(self):
        return self.message
