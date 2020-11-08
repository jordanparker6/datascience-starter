import numpy as np

class Metrics:

    def mape(self, y, yhat):
        return np.mean(np.abs((y - yhat) / y)) * 100
    
