import pymc3 as pm
import numpy as np
from sklearn.base import BaseEstimator

class PyMC3Estimator(BaseEstimator):
    """
    PyMC3Estimator
      A base class for PyMC3 estimators using the sklearn API.
    """
    def __init__(self):
        self.model = pm.Model()

    def fit(self, X, y, samples=1000, tune=1000, **kwargs):
        with self.model as model:
            self.definition(model, X, y, **kwargs)
            self.map = pm.find_MAP()
            self.trace = pm.sample(samples, tune=tune, progressbar=True, **kwargs)
    
    def definition(self, model, X, y):
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