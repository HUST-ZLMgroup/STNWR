import numpy as np


def get_AICc(model):
    n = model.n
    k = model.tr_S
    # sigma2 = model.sigma2
    aicc = -2.0 * model.llf + 2.0 * n * (k + 1.0) / (n - k - 2.0)
    # equivalent to below but can't control denominator of sigma without altering GLM familt code
    # aicc = n*np.log(sigma2) + n*np.log(2.0*np.pi) + n*(n+k)/(n-k-2.0)
    return aicc


def get_AIC(model):
    k = model.tr_S
    aic = -2.0 * model.llf + 2.0 * (k + 1)
    return aic


def get_BIC(model):
    n = model.n  # (scalar) number of observations
    k = model.tr_S
    bic = -2.0 * model.llf + (k + 1) * np.log(n)
    return bic


def get_CV(model):
    # aa = model.resid_response.reshape((-1, 1)) / (1.0 - model.influ)
    # cv = np.sum(aa ** 2) / model.n
    cv = np.sum(model.resid_response ** 2) / model.n
    return cv
