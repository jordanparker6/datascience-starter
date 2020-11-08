import pymc3 as pm
import numbers
import numpy as np
import theano.tensor as tt
from datascience_starter.base.estimators import PyMC3Estimator
from datascience_starter.models.glm.families import families

class GLM(PyMC3Estimator):
    """
    GLM
     A bayesian implempentation of Generalized Linear Models.

     A GLM is a generalised approach to linear models that can model
     any likelihood in the exponential family using a linear combination
     of the depedent variables and a link function.

     -> Logisitic Regression is a bernouli liklihood function
     -> Linear Regression is a normal liklihood function
    """

    def __init__(self, 
            likelihood, 
            prior = pm.Laplace, 
            prior_params = { "mu": 0,  "b": 1 }
        ):
        super().__init__()
        assert likelihood in families.keys(), "Likelihood not in expoential family."
        self.family = families[likelihood]
        self.prior = prior
        self.params = prior_params

    def definition(self, model, X, y):
        # build assertion to check shape of X and y
        with model:
            # Priors for linear regression
            alpha = self.prior('alpha', **self.params)
            beta = self.prior('beta', shape=(X.shape[1]), **self.params)

            # Priors for likelihood family
            priors = {}
            for key, val in self.family.priors.items():
                if isinstance(val, (numbers.Number, np.ndarray, np.generic)):
                    priors[key] = val
                else:
                    priors[key] = model.Var(key, val)

            # Likelihood Priors
            yhat = self.family.link(alpha + pm.math.dot(X, beta))
            priors[self.family.parent] = yhat
            self.family.likelihood("y", observed=y, **priors)