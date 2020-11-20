import pymc3 as pm
import theano.tensor as tt

sigmoid = pm.math.sigmoid
softmax = tt.nnet.softmax
exp = pm.math.exp
identity = lambda x: x
inverse = lambda x: x ** -1 
log = pm.math.log

class Family:
    """Base class for distributions from the exponential family
    """
    link = None         #: Mean link function
    likelihood = None   #: Likelihood distribution
    parent = None       #: Likelihood parameter predicted by GLM
    priors = {}         #: Likelihood priors

class Normal(Family):
    link = identity
    likelihood = pm.Normal
    parent = "mu"
    priors = {
        "sigma": pm.HalfCauchy.dist(beta=10, testval=1.0)
    }

class LogNormal(Family):
    link = exp
    likelihood = pm.Normal
    parent = "mu"
    priors = {
        "sigma": pm.HalfCauchy.dist(beta=10, testval=1.0)
    }

class StudentT(Family):
    link = identity
    likelihood = pm.StudentT
    parent = "mu"
    priors = { 
        "sigma": pm.HalfCauchy.dist(beta=10, testval=1.0), 
        "nu": 1
    }

class Bernoulli(Family):
    link = sigmoid
    likelihood = pm.Bernoulli
    parent = "p"

class Binomial(Family):
    link = sigmoid
    likelihood = pm.Binomial
    parent = "p"
    priors = {
        "n": 1
    }

class Categorical(Family):
    link = softmax
    likelihood = pm.Categorical
    parent = "p"

class Poisson(Family):
    link = exp
    likelihood = pm.Poisson
    parent = "mu"
    priors = {
        "mu": pm.HalfCauchy.dist(beta=10, testval=1.0)
    }

class NegativeBinomial(Family):
    link = exp
    likelihood = pm.NegativeBinomial
    parent = "mu"
    priors = {
        "mu": pm.HalfCauchy.dist(beta=10, testval=1.0),
        "alpha": pm.HalfCauchy.dist(beta=10, testval=1.0),
    }

class Gamma(Family):
    link = exp
    likelihood = pm.Gamma
    parent = "mu"
    priors = {
        "sigma": pm.HalfCauchy.dist(beta=10, testval=1.0)
    }

families = {
    "normal": Normal,
    "log_normal": LogNormal,
    "student-t": StudentT,
    "poisson": Poisson,
    "bernoulli": Bernoulli,
    "binomial": Binomial,
    "categorical": Categorical,
    "negative_binomial": NegativeBinomial,
    "gamma": Gamma
}