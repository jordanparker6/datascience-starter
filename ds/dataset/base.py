from typing import Optional
from ds.core.base import Base
from ds.processing import ProcessorPipeline
from sklearn.model_selection import KFold

class Dataset(Base):
    """
    Dataset is bundle of train / test data and dataset metadata.
    
    The dataset receives dataset metadata (e.g. labels, categorical features)
    and a processor class for feature engineering. It is serialised and loaded
    for preprocessing / feature engineering post training.
    """
    
    train = None
    test = None
    
    def __init__(self, label: str, processor: Optional[ProcessorPipeline] = None):
        self.label = label
        self.processor = processor
    
    @classmethod
    @abstractmethod
    def load(cls):
        raise NotImplementedError
    
    def process(self, df, mode: str = "predict", processor: Optional[ProcessorPipeline] = None):
        self.processor = processor if processor else self.processor
        if not self.processor:
            assert "A processor must be attached to the dataset."
        if mode == "train":
            df = self.processor.fit(df)
            y = df[self.label]
            X = df.drop(columns=[self.label])
            self.train = (X, y)
        elif mode == "eval":
            df = self.processor.transform(df, train=True)
            y = df[self.label]
            X = df.drop(columns=[self.label])
            self.test = (X, y)
        elif mode == "predict":
            df = self.processor.transform(df, train=False)
            X = df
            y = None
        else:
            raise "Mode must be one of train, eval or predict."
        return X, y
    
    def inverse_transform_predictions(self, pred):
        return self.processor.inverse_transform(pred, transform_features=False)
    
    @classmethod
    def from_splits(cls, folds=5, **kwargs):
        df = cls.load()
        kf = KFold(n_splits=5)
        for i, indexes in enumerate(kf.split(df)):
            dataset = cls(**kwargs)
            train_index, test_index = indexes
            train_df, test_df = df.iloc[train_index, :], df.iloc[test_index, :]
            dataset.process(train_df, mode="train")
            dataset.process(test_df, mode="eval")
            yield dataset
    
    @classmethod
    def from_training_only(cls, **kwargs):
        df = cls.load()
        dataset = cls(**kwargs)
        dataset.process(df, mode="train")
        dataset.test = dataset.train
        return dataset
    
    def save(self, path, with_data=False):
        if not with_data:
            train = None
            test = None
        log.info(f"Saving Dataset to Path: {path}")
        joblib.dump(self, path)