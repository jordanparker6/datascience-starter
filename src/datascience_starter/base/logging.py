import os
import logging

logging.basicConfig(
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s', 
    level=os.environ.get("LOGLEVEL", "INFO")
)

class Logger:
    """The default logging class.
    Attributes:
        log (logging.Logger): An initiated logging object.
    """
    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(os.environ.get("LOGLEVEL", "INFO"))