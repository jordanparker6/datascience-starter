"""An implementation of the Processor interface and ProcessorPipeline"""
from abc import abstractmethod
from typing import List, Optional
import pandas as pd
from starter_pack.core.base import Base

class ProcessBase(Base):
    """A base class to implement a processor interface."""
    def __init__(self, columns: List[str]):
        self.columns = columns
    
    @abstractmethod
    def fit(self, df: pd.DataFrame):
        raise NotImplementedError
    
    @abstractmethod
    def transform(self, df: pd.DataFrame):
        raise NotImplementedError
    
    @abstractmethod
    def inverse_transform(self, df: pd.DataFrame):
        raise NotImplementedError
    
#####################################################
### PROCESSOR PIPELINE CLASS ########################
#####################################################
    
class ProcessorPipeline(Base):
    """
    A base class to implement a pipeline of feature engineering steps.
    
    A processor pipeline will receive a list of Processors and execute each processor sequentially
    updating the underlying dataframe with the transformed values. A seperate pipeline is applied to
    the target and feature variables.
    """
    def __init__(self, feature_steps: List[Optional[ProcessBase]] = [], target_steps: List[Optional[ProcessBase]] = []):
        self.feature_steps = feature_steps
        self.target_steps = target_steps
        
    def add(self, step: ProcessBase, feature=True):
        if feature:
            self.feature_steps.append(step)
        else: 
            self.target_steps.append(step)
        return self
        
    def fit(self, df: pd.DataFrame):
        """Fits and transforms all features and target steps."""
        for step in self.feature_steps:
            df = step.fit(df)
        for step in self.target_steps:
            df = step.fit(df)
        return df
    
    def transform(self, df: pd.DataFrame, train=True):
        """Transforms all features using a prefited processor. If used in a training loop, also transforms targets."""
        for step in self.feature_steps:
            df = step.transform(df)
        if train:
            for step in self.target_steps:
                df = step.fit(df)
        return df
    
    def inverse_transform(self, df: pd.DataFrame, transform_features: bool = False):
        if transform_features:
            for step in self.feature_steps[::-1]:
                df = step.inverse_transform(df)
        for step in self.target_steps[::-1]:
            df = step.inverse_transform(df)
        return df