from pathlib import Path
import sys

class CustomException(Exception):
    def __init__(self, error_name, error_detail):
        super().__init__(error_name)
        self.error_name = error_name
        _, _, exc_tb = error_detail.exc_info()
        file_name = Path(exc_tb.tb_frame.f_code.co_filename).name
        line_number = exc_tb.tb_lineno
        self.error_message = (
                                f"Error occurred in Python script name "
                                f"[{file_name}] line number [{line_number}] "
                                f"error message [{error_name}]"
                            )
    def __str__(self):
            return self.error_message


