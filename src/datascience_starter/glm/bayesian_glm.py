import numpy as np
import pymc3 as pm
from sklearn.base import BaseEstimator

def sigmoid(x):
    return 1 / ( 1 + np.exp(-x))

link_functions = {
    "normal": lambda x: x,
    "student-t": lambda x: x,
    "exponential": lambda x: -1 * np.power(x, -1),
    "gamma": lambda x: -1 * np.power(x, -1),
    "poisson": np.exp,
    "bernoulli": sigmoid,
    "binomial": sigmoid,
    "categorical": sigmoid,
    "multinomial": sigmoid
}

class GLM(BaseEstimator):
    """
    GLM
     A bayesian implempentation of Geneeralized Linear Models.

     A GLM is a generalised approach to linear models that can model
     any likelihood in the exponential family using a linear combination
     of the depedent variables and a link function.

     -> Logisitic Regression is a bernouli liklihood function
    """

    def __init__(self, likelihood, partial_pooled=False):
        assert likelihood in link_functions.keys(), "Liklihood not in expoential family"
        self.dist = dist
        self.partial_pooled = partial_pooled
        self.link_function = link_functions[likelihood]
        self.model = pm.Model()

    def definition(self, model, X, y):

        with model:

            # Priors for unknown model parameters
            alpha = pm.Normal('alpha', mu=0, sd=10)
            beta = pm.Normal('beta', mu=0, sd=10, shape=X.shape[1])

            # Expected value of outcome
            mu = self.link_funcation(alpha + beta * X)

            # Likelihood function
            if dist == 'bernoulli':
                y = pm.Bernoulli('y', p=mu, observed=y)
            elif dist == 'categorical':
                y = pm.Bernoulli('y', p=mu, observed=y)
            elif dist == 'binomial':
                raise NotImplementedError
            elif dist == 'multinomial':
                raise NotImplementedError
            elif dist == 'poisson':
                y = pm.Poisson('y', mu=mu, observed=y)
            elif dist == 'normal':
                sigma = pm.HalfCauchy.dist(beta=10, testval=1.0)
                y = pm.Normal('y', mu=mu, sigma=sigma)
            elif dist == 'student-t':
                sigma = pm.HalfCauchy.dist(beta=10, testval=1.0)
                nu = pm.InverseGamma("nu", alpha=1, beta=1)
                y = pm.StudentT('y', mu=mu, sigma=sigma, nu=nu)
            elif dist == 'gamma':
                raise NotImplementedError
            elif dist == 'exponential':
                raise NotImplementedError
    
    def fit(self, X, y):
        with self.model as model:
            self.definition(model, X, y)
            self.trace = pm.sample(tune=5000, draws=500, chains=4, progressbar=True)


    def plot_trace(self):
        pm.traceplot(self.trace)

    def plot_posterior(self):
        pm.plot_posterior(self.trace)