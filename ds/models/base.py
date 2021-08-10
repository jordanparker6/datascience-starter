from ds.core.base import Base
import shap
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score

METRICS = {
    "r2": r2_score,
    "mse": mean_squared_error,
    "mape": mean_absolute_percentage_error,
    "mae": mean_absolute_error
}

class BaseModel(Base):
    """
    Provides an interface for model training nad predictions
    while implementing shapely values for explanability
    """
    def __init__(self, eval_metric: str):
        self._model = None
        self.metrics = METRICS
        self.eval_metric = eval_metric
        self.loss = METRICS[eval_metric]
    
    @abstractmethod
    def forward(self, X: np.ndarray):
        raise NotImplementedError
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError
        
    def evaluate(self, X: np.ndarray, y: np.ndarray):
        yhat = self.forward(X)
        metrics = { k: self.metrics[k](y, yhat) for k, v in self.metrics.items() }
        log.info(metrics)
        return metrics
        
    def predict(self, X: np.ndarray):
        return self.forward(X)
    
    def explain(self, 
            X_train: np.ndarray, X_test: np.ndarray, nsamples=100, 
            show_waterfall=True, show_force=True, show_bar=True, show_beeswarm=True
        ):
        X_train, X_test = X_train.sample(nsamples), X_test.sample(nsamples)
        self.explainer = shap.KernelExplainer(self.predict, X_train, nsamples=nsamples)
        self.shap_values = self.explainer.shap_values(X_test, nsamples=nsamples)
        if show_bar:
            shap.summary_plot(self.shap_values, X_test, max_display=500)
        #if show_beeswarm:
            #shap.summary_plot(self.shap_values, X_train)
        #if show_force:
            #shap.plots.force(self.explainer.expected_value[0], self.shap_values[0], X_train)
        #if show_waterfall:
            #shap.plots.waterfall(self.explainer.expected_value[0], self.shap_values[0], X_train)

class SKLearnModel(BaseModel):
    """
    A wrapper for SKLearn models to implement the BaseModel interface
    """
    def __init__(self, model, eval_metric: str, **kwargs):
        super().__init__(eval_metric)
        self._model = model(**kwargs)
        
    def forward(self, X: np.ndarray):
        return self._model.predict(X)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self._model.fit(X, y)


class PyMC3ModelBase(BaseModel):
    """
    A base class for PyMC3 estimators using the BaseModel interface.
    
    Attributes:
        model (pymc3.Model): A PyMC3 model object.
        map: The Maxiumum A Posterior estimate of the parameter values.
        trace: The sampled values from the posterior distributions.
    """
    def __init__(self, eval_metric: str):
        super.__init__(eval_metric)
        self.model = pm.Model()
        self.map = None
        self.trace = None

    def fit(self, X: np.ndarray, y: np.ndarray, samples: int = 1000, tune: int = 1000, **kwargs):
        """Defines the PyMC3 model and evaluates the trace and MAP.

        Args:
            X: The depedent variables / features.
            y: The independent variable / target.
            samples (optional): The numper of samples to draw using HMCM.
            tune (optional): The number of samples to burn-in during HMCM.
            **kwargs: Additional key word arguements for PyMC3's pm.sample method.
        
        """
        with self.model as model:
            self._definition(model, X, y, **kwargs)
            self.map = pm.find_MAP()
            self.trace = pm.sample(samples, tune=tune, progressbar=True, **kwargs)
        return self
    
    @abstractmethod
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

    def forward(self, X, **kwargs):
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

    def plot_graph(self):
        return pm.model_to_graphviz(self.model)

    def plot_autocorr(self):
        pm.plots.autocorrplot(trace=self.trace)
    
    def plot_energy(self):
        bfmi = np.max(pm.stats.bfmi(self.trace))
        max_gr = max(np.max(gr_stats) for gr_stats in pm.stats.rhat(self.trace).values()).values
        (pm.energyplot(self.trace, legend=False, figsize=(6, 4)).set_title("BFMI = {}\nGelman-Rubin = {}".format(bfmi, max_gr)))