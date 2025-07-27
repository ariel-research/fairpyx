import logging
from io import StringIO
import re
import ast


class LogCaptureHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.log_capture_string = StringIO()
        self.setFormatter(logging.Formatter('%(message)s'))

    def emit(self, record):
        message = self.format(record)
        if "final budget b*" in message or "final prices p*" in message:
            self.log_capture_string.write(message + '\n')

    def get_logs(self):
        return self.log_capture_string.getvalue()

    def extract_prices(self):
        """
        Extracts the prices from the captured logs.

        :return: A dictionary containing the final prices.
        :rtype: dict or None
        """
        logs = self.get_logs().split('\n')
        prices_pattern = r"final prices p\* = ({.*})"

        for log in logs:
            match = re.search(prices_pattern, log)
            if match:
                prices_str = match.group(1)
                # Convert the string representation of the dictionary to an actual dictionary
                prices = ast.literal_eval(prices_str)
                return prices
        return None