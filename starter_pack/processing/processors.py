"""A collection of implementations of ProcessorBase"""
from typing import List, Callable
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder as SKLearnOneHotEncoder
from starter_pack.processing.base import ProcessBase
from starter_pack.io import AsyncFetch

class DropColumns(ProcessBase):
    """Drops columns from the dataset."""
    def fit(self, df: pd.DataFrame):
        self._dropped = df[[col for col in df.columns if col not in self.columns]]
        return df.drop(columns=self.columns, errors='ignore')
    
    def transform(self, df: pd.DataFrame):
        return self.fit(df)
    
    def inverse_transform(self, df: pd.DataFrame):
        return pd.concat(df, self._dropped, axis=1)

class ReplaceNaN(ProcessBase):
    """Replaces all NaN in the dataset."""
    def __init__(self, value: int = 0):
        self.value = 0
        
    def fit(self, df: pd.DataFrame):
        return df.fillna(self.value)
    
    def transform(self, df: pd.DataFrame):
        return self.fit(df)
    
    def inverse_transform(self, df: pd.DataFrame):
        self.log.warning("ReplaceNaN.inverse_transform() currently doesn't support an inverse_transfrom. Original DataFrame returned")
        return df
        
    
class OneHotEncode(ProcessBase):
    """OneHotEncodes categorical variables."""
    def __init__(self, columns, **kwargs):
        super().__init__(columns)
        self.encoder = SKLearnOneHotEncoder(**kwargs, sparse=False)
        
    def fit(self, df: pd.DataFrame):
        X = self.encoder.fit_transform(df[self.columns])
        self._new_col_names = self.encoder.get_feature_names(self.columns)
        df = df.drop(columns=self.columns)
        new_df = pd.DataFrame(X, columns=self._new_col_names, index=df.index)
        df = pd.concat((df, new_df), axis=1)
        return df
    
    def transform(self, df: pd.DataFrame):
        X = self.encoder.transform(df[self.columns])
        df = df.drop(columns=self.columns)
        new_df = pd.DataFrame(X, columns=self._new_col_names, index=df.index)
        df = pd.concat((df, new_df), axis=1)
        return df
    
    def inverse_transform(self, df):
        X = self.encoder.inverse_transfrom(df[self._new_col_names])
        df = df.drop(columns=self._new_col_names)
        df[self.columns] = X
        return df

class SKLearnProcessor(ProcessBase):
    """A wrapper for any SKLearn processing objects."""
    def __init__(self, sklearn_class, columns: List[str]):
        super().__init__(columns)
        self._class = sklearn_class
  
        
    def fit(self, df: pd.DataFrame):
        df[self.columns] = self._class.fit_transform(df[self.columns])
        return df
    
    def transform(self, df: pd.DataFrame):
        df[self.columns] = self._class.transform(df[self.columns])
        return df
    
    def inverse_transform(self, df: pd.DataFrame):
        df[self.columns] = self._class.inverse_transform(df[self.columns])
        return df

class DatetimeEncoder(ProcessBase):
    def __init__(self, columns: List[str], min_freq: str = "month"):
        super().__init__(columns)
        self.min_freq = min_freq
        
    def fit(self, df: pd.DataFrame):
        for col in self.columns:
            df = self.sinusoidal_position_encoding(df, col)
        return df
    
    def transform(self, df: pd.DataFrame):
        return self.fit(df)
    
    def inverse_transform(self, df: pd.DataFrame):
        raise NotImplementedError
    
    def sinusoidal_position_encoding(self, df: pd.DataFrame, col: str):
        """
        Encodes the position of hour, day, month seaonality. RBF could be used in place.
        """
        hour = df[col].dt.hour / 24
        day = df[col].dt.day / 30.5
        month = df[col].dt.month / 12
        year = df[col].dt.year
        if self.min_freq in ['hour']:
            df[f'{col}_sin_hour'] = np.sin(2 * np.pi * hour)
            df[f'{col}_cos_hour'] = np.cos(2 * np.pi * hour)
        if self.min_freq in ["hour", "day"]:
            df[f'{col}_sin_day'] = np.sin(2 * np.pi * day)
            df[f'{col}_cos_day'] = np.cos(2 * np.pi * day)
        if self.min_freq in ["hour", "day", "month"]:
            df[f'{col}_sin_month'] = np.sin(2 * np.pi * month)
            df[f'{col}_cos_month'] = np.cos(2 * np.pi * month)
        df[f'{col}_year'] = year
        df = df.drop(columns=[col])
        return df

class Sentence2Vec(ProcessBase):
    """A huggingface sentence encoder leveraing a Transformers model."""
    def __init__(self, columns: List[str], model:str = "paraphrase-MiniLM-L3-v2"):
        #self.model = SentenceTransformer(model)
        self.model = None
        self.columns = columns
    
    def fit(self, df):
        for col in self.columns:
            X = self.model.encode(df[col])
            new_df = pd.DataFrame(X, columns=[f"{co}_emb_{i}" for i in range(len(X))])
            df = df.drop(columns=[col])
            df = pd.concat((df, new_df), axis=1)
        return df
    
    def transform(self, df):
        return self.fit(df)

    def inverse_transform(self, df: pd.DataFrame):
        self.log.warning("Sentence2Vec.inverse_transform() currently doesn't support an inverse_transfrom. Original DataFrame returned")
        return df

class Transform(ProcessBase):
    """A class to apply arbitary transformations to pandas columns"""
    def __init__(self, func: Callable, suffix: str, columns: List[str]):
        self.func = func
        self.suffix = suffix
        self.columns = columns
    
    def fit(self, df: pd.DataFrame):
        for col in self.columns:
            df[f"{col}_{self.suffix}"] = df[col].apply(self.func)
        return df

    def transform(self, df: pd.DataFrame):
        return self.fit(df)
    
    def inverse_transform(self, df: pd.DataFrame):
        dropcols = [f"{col}_{self.suffix}" for col in self.columns]
        df = df.drop(columns=dropcols)
        return df