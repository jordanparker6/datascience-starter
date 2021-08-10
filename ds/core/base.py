"""A base class for all serialisable classes"""
from abc import ABC, abstractmethod
import logging
import joblib

log = logging.getLogger(_name_)

class Base(ABC):
    """A Base component for all class objects."""
    @classmethod
    def from_pickle(cls, path: str):
        log.info(f"Loading Dataset from Path: {path}")
        return joblib.load(path)
    
    def save(self, path: str):
        log.info(f"Saving Dataset to Path: {path}")
        joblib.dump(self, path)