import pymc3 as pm
import numpy as np
from sklearn.base import BaseEstimator

class PyMC3Estimator:
    """
    A base class for PyMC3 estimators using the sklearn API.
    
    Attributes:
        model (pymc3.Model): A PyMC3 model object.

    """

    def __init__(self):
        self.model = pm.Model()

    def fit(self, X: np.ndarray, y: np.ndarray, samples: int = 1000, tune: int = 1000, **kwargs):
        """Defines the PyMC3 model and evaluates the trace and MAP.

        Args:
            X: The depedent variables / features.
            y: The independent variable / target.
            samples (optional): The numper of samples to draw using HMCM.
            tune (optional): The number of samples to burn-in during HMCM.
            **kwargs: Additional key word arguements for PyMC3's pm.sample method.
        
        Attributes:
            map: The Maxiumum A Posterior estimate of the parameter values.
            trace: The sampled values from the posterior distributions.
            
        """
        with self.model as model:
            self._definition(model, X, y, **kwargs)
            self.map = pm.find_MAP()
            self.trace = pm.sample(samples, tune=tune, progressbar=True, **kwargs)
        return self
    
    def _definition(self, model, X, y):
        """The definition of the PyMC3 model.

        Args:
            model (pymc3.Model): The PyMC3 model to define (e.g. self.model).
            X (np.ndarray): A matrix of the feactures.
            y (np.ndarray): The target vector.

        Raises:
            NotImplementedError: must be defined in inherited class.
        """
        raise NotImplementedError

    def predict(self, X, **kwargs):
        raise NotImplementedError

    def sample_posterior_predictive(self, samples):
        return pm.sample_posterior_predictive(self.trace, samples=samples, model=self.model)

    def summary(self):
        return pm.summary(self.trace)

    def plot_priors(self):
        raise NotImplementedError

    def plot_trace(self):
        pm.traceplot(self.trace)

    def plot_posterior(self):
        pm.plot_posterior(self.trace)

    def plot_joint(self):
        pm.plot_joint(self.trace, kind='kde', fill_last=False)

    def plot_graph(self):
        return pm.model_to_graphviz(self.model)

    def plot_autocorr(self):
        pm.plots.autocorrplot(trace=trace)
    
    def plot_energy(self):
        bfmi = np.max(pm.stats.bfmi(self.trace))
        max_gr = max(np.max(gr_stats) for gr_stats in pm.stats.rhat(self.trace).values()).values
        (pm.energyplot(self.trace, legend=False, figsize=(6, 4)).set_title("BFMI = {}\nGelman-Rubin = {}".format(bfmi, max_gr)))