import pymc3 as pm
import numbers
import numpy as np
import theano.tensor as tt
from datascience_starter.base.estimators import PyMC3Estimator
from datascience_starter.models.glm.families import families

class GLM(PyMC3Estimator):
    """A bayesian implempentation of Generalized Linear Models.

     A GLM is a generalised approach to linear models that can model
     any likelihood in the exponential family using a linear combination
     of the depedent variables and a link function.

    Args:
        liklihood: A string indicatin the distribution of the liklihood.
        prior: A PyMC3 distribution for the priors over alpha and beta.
        prior_params: The parameters for the priors over alpha and beta.

    """

    def __init__(self, 
            likelihood: str, 
            prior = pm.Laplace, 
            prior_params: Dict(str, float) = { "mu": 0.0,  "b": 1.0 }
        ):
        super().__init__()
        assert likelihood in families.keys(), "Likelihood not in expoential family."
        self.family = families[likelihood]
        self.prior = prior
        self.params = prior_params

    def definition(self, model, X: np.ndarray, y: np.ndarray):
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