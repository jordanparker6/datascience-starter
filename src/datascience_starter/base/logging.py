import logging, os

logging.basicConfig(
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s', 
    level=os.environ.get("LOGLEVEL", "INFO")
)

class Logger:
    """A base class for logging.

    Attributes:
        log (obj): A logger object to write logs to.
        
    """

    def __init__(self):
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(os.environ.get("LOGLEVEL", "INFO"))