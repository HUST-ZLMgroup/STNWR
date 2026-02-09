from scipy.stats import t
from spglm.family import Gaussian
from .diagnostics import get_AIC, get_BIC
from .summary import *


class GTWRResults(object):
    def __init__(self, model, params, predy, influ, CCT):
        self.model = model
        self.n = model.n
        self.y = model.y.flatten()
        self.X = model.X
        self.k = model.k
        self.params = params
        self.predy = predy.flatten()
        self.influ = influ

        self.resid_response = self.y - self.predy
        self.resid_ss = np.sum(self.resid_response.flatten() ** 2)
        self.tr_S = np.sum(self.influ)
        self.ENP = self.tr_S
        self.df_model = self.n - self.tr_S
        self.sigma2 = self.resid_ss / (self.n - self.tr_S)
        self.scale = self.sigma2
        self.llf = Gaussian().loglike(self.y, self.predy, scale=self.scale)
        self.aic = get_AIC(self)
        self.aicc = get_AICc(self)
        self.bic = get_BIC(self)
        self.cv = get_CV(self)
        self.R2 = 1.0 - self.resid_ss / np.sum((self.y - np.mean(self.y)) ** 2)
        self.adj_R2 = 1. - (1. - self.R2) * (self.n - 1) / (self.n - self.ENP - 1)
        self.adj_alpha = (np.array([.1, .05, .001]) * self.k) / self.ENP
        self.critical_tval = t.ppf(1 - np.abs(self.adj_alpha[1]) / 2.0, self.n - 1)
        self.CCT = CCT * self.sigma2
        self.bse = np.sqrt(self.CCT)
        self.tvalues = self.params / self.bse
        self.pvalues = t.sf(np.abs(self.tvalues), self.n - 1) * 2

        self.summary = summaryModel(self) + summaryGLM(self) + summaryGTWR(self)


class GTWRResultsLite(object):
    def __init__(self, model, resid, influ, params):
        self.y = model.y
        self.n = model.n
        self.influ = influ
        self.resid_response = resid
        self.params = params

        self.tr_S = np.sum(self.influ)
        self.predy = self.y - self.resid_response
        self.llf = Gaussian().loglike(self.y, self.predy)
        self.resid_ss = np.sum(self.resid_response ** 2)


class MGTWRResults(object):
    def __init__(self, model, params, predy, CCT, ENP_j):
        self.model = model
        self.n = model.n
        self.y = model.y.flatten()
        self.X = model.X
        self.k = model.k
        self.params = params
        self.predy = predy.flatten()
        self.ENP_j = ENP_j

        self.tr_S = np.sum(self.ENP_j)

        self.resid_response = self.y - self.predy
        self.resid_ss = np.sum(self.resid_response.flatten() ** 2)
        self.ENP = self.tr_S
        self.df_model = self.n - self.tr_S
        self.sigma2 = self.resid_ss / (self.n - self.tr_S)
        self.scale = self.sigma2
        self.llf = Gaussian().loglike(self.y, self.predy, scale=self.scale)
        self.aic = get_AIC(self)
        self.aicc = get_AICc(self)
        self.bic = get_BIC(self)
        self.cv = get_CV(self)
        self.R2 = 1.0 - self.resid_ss / np.sum((self.y - np.mean(self.y)) ** 2)
        self.adj_R2 = 1. - (1. - self.R2) * (self.n - 1) / (self.n - self.ENP - 1)
        self.CCT = CCT * self.sigma2
        self.bse = np.sqrt(self.CCT)
        self.tvalues = self.params / self.bse
        self.pvalues = t.sf(np.abs(self.tvalues), self.n - 1) * 2
        self.adj_alpha_j = np.array([.1, .05, .001]) / np.array(self.ENP_j).reshape((-1, 1))

        self.summary = summaryModel(self) + summaryGLM(self) + summaryMGTWR(self)

    def critical_tval(self, alpha=None):
        n = self.n
        if alpha is not None:
            alpha = np.abs(alpha) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        else:
            alpha = np.abs(self.adj_alpha_j[:, 1]) / 2.0
            critical = t.ppf(1 - alpha, n - 1)
        return critical
