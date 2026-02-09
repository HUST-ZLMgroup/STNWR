import numpy as np
from spglm.glm import GLM
from .diagnostics import get_AICc, get_CV


def summaryModel(self):
    summary = f'{"=":=^80}\n'
    summary += f'{"Model type:":>42}  {"Gaussian":<}\n'
    summary += f'{"Number of observations:":>42}  {self.n:<}\n'
    summary += f'{"Number of covariates:":>42}  {self.k:<}\n\n'
    return summary


def summaryGLM(self):
    XNames = ["X" + str(i) for i in range(self.k)]
    glm_rslt = GLM(self.model.y, self.model.X, constant=False).fit()

    summary = f'{"Global Regression Results":^80}\n'
    summary += f'{"-":-^80}\n'

    summary += f'{"Residual sum of squares:":>42}  {glm_rslt.deviance:<.3f}\n'
    summary += f'{"Log-likelihood:":>42}  {glm_rslt.llf:<.3f}\n'
    summary += f'{"AIC:":>42}  {glm_rslt.aic:<.3f}\n'
    summary += f'{"AICc:":>42}  {get_AICc(glm_rslt):<.3f}\n'
    summary += f'{"BIC:":>42}  {glm_rslt.bic:<.3f}\n'
    summary += f'{"CV:":>42}  {get_CV(glm_rslt):<.3f}\n'
    summary += f'{"R2:":>42}  {glm_rslt.D2:<.3f}\n'
    summary += f'{"Adjusted R2:":>42}  {glm_rslt.adj_D2:<.3f}\n'

    summary += f'{"Variable":^16}{"Est.":^16}{"SE":^16}{"t(Est/SE)":^16}{"p-value":^16}\n'
    summary += f'{"-":-<16} {"-":-^15} {"-":-^15} {"-":-^15} {"-":->15}\n'
    for i in range(self.k):
        summary += (f'{XNames[i]:^16}{glm_rslt.params[i]:^16.3f}{glm_rslt.bse[i]:^16.3f}'
                    f'{glm_rslt.tvalues[i]:^16.3f}{glm_rslt.pvalues[i]:^16.3f}\n')
    summary += "\n"
    return summary


def summaryGTWR(self):
    XNames = ["X" + str(i) for i in range(self.k)]
    if self.model.tau == 0:
        summary = f'{"Geographically Weighted Regression (GWR) Results":^80}\n'
    else:
        summary = f'{"Geographically and Temporally Weighted Regression (GTWR) Results":^80}\n'
    summary += f'{"-":-^80}\n'

    if self.model.fixed:
        summary += f'{"Spatial kernel:":>42}  {"Fixed " +  self.model.kernel:<}\n'
    else:
        summary += f'{"Spatial kernel:":>42}  {"Adaptive " +  self.model.kernel:<}\n'

    summary += f'{"Bandwidth used:":>42}  {self.model.bw:<}\n'
    summary += f'{"tau used:":>42}  {self.model.tau:<}\n\n'

    summary += f'{"Diagnostic information":^80}\n'
    summary += f'{"-":-^80}\n'

    summary += f'{"Residual sum of squares:":>42}  {self.resid_ss:<.3f}\n'
    summary += f'{"Effective number of parameters (trace(S)):":>42}  {self.tr_S:<.3f}\n'
    summary += f'{"Degree of freedom (n - trace(S)):":>42}  {self.df_model:<.3f}\n'
    summary += f'{"Sigma estimate:":>42}  {np.sqrt(self.sigma2):<.3f}\n'
    summary += f'{"Log-likelihood:":>42}  {self.llf:<.3f}\n'
    summary += f'{"AIC:":>42}  {self.aic:<.3f}\n'
    summary += f'{"AICc:":>42}  {self.aicc:<.3f}\n'
    summary += f'{"BIC:":>42}  {self.bic:<.3f}\n'
    summary += f'{"CV:":>42}  {self.cv:<.3f}\n'
    summary += f'{"R2:":>42}  {self.R2:<.3f}\n'
    summary += f'{"Adjusted R2:":>42}  {self.adj_R2:<.3f}\n'
    summary += f'{"Adj. alpha (95%):":>42}  {self.adj_alpha[1]:<.3f}\n'
    summary += f'{"Adj. critical t value (95%):":>42}  {self.critical_tval:<.3f}\n\n'

    summary += f'{"Summary Statistics For GTWR Parameter Estimates":^80}\n'
    summary += f'{"-":-^80}\n'

    summary += f'{"Variable":^15}{"Mean":^13}{"STD":^13}{"Min":^13}{"Median":^13}{"Max":^13}\n'
    summary += f'{"-":-<15} {"-":-^12} {"-":-^12} {"-":-^12} {"-":-^12} {"-":->12}\n'

    for i in range(self.k):
        summary += (f'{XNames[i]:^15}{np.mean(self.params[:, i]):^13.3f}{np.std(self.params[:, i]):^13.3f}'
                    f'{np.min(self.params[:, i]):^13.3f}{np.median(self.params[:, i]):^13.3f}'
                    f'{np.max(self.params[:, i]):^13.3f}\n')
    summary += f'{"=":=^80}\n'

    return summary


def summaryMGTWR(self):
    XNames = ["X" + str(i) for i in range(self.k)]
    if np.all(self.model.bws_taus[:, 1] == 0):
        summary = f'{"Multi-Scale Geographically Weighted Regression (MGWR) Results":^80}\n'
    else:
        summary = f'{"Multi-Scale Geographically and Temporally Weighted Regression (MGTWR) Results":^80}\n'
    summary += f'{"-":-^80}\n'

    if self.model.fixed:
        summary += f'{"Spatial kernel:":>42}  {"Fixed " + self.model.kernel:<}\n'
    else:
        summary += f'{"Spatial kernel:":>42}  {"Adaptive " + self.model.kernel:<}\n'

    summary += f'{"Criterion for optimal bandwidth:":>42}  {self.model.selector.criterion:<}\n'

    if self.model.selector.rss_score:
        summary += f'{"Score of Change (SOC) type:":>42}  {"RSS":<}\n'
    else:
        summary += f'{"Score of Change (SOC) type:":>42}  {"Smoothing f":<}\n'

    summary += f'{"Termination criterion for MGTWR:":>42}  {self.model.selector.tol_multi:<}\n\n'

    summary += f'{"MGTWR bandwidths and taus":^80}\n'
    summary += f'{"-":-^80}\n'

    summary += (f'{"Variable":^11}{"Bandwidth":^13}{"tau":^13}'
                f'{"ENP_j":^13}{"Adj t-val(95%)":^15}{"Adj alpha(95%)":^15}\n')
    summary += f'{"-":-<11} {"-":-^12} {"-":-^12} {"-":-^11} {"-":-^14} {"-":->15}\n'

    for j in range(self.k):
        summary += (f'{XNames[j]:^11}{self.model.bws_taus[j][0]:^13.3f}{self.model.bws_taus[j][1]:^13.3f}'
                    f'{self.ENP_j[j]:^13.3f}{self.critical_tval()[j]:^15.3f}'
                    f'{self.adj_alpha_j[j, 1]:^15.3f}\n')
    summary += "\n"

    summary += f'{"Diagnostic information":^80}\n'
    summary += f'{"-":-^80}\n'

    summary += f'{"Residual sum of squares:":>42}  {self.resid_ss:<.3f}\n'
    summary += f'{"Effective number of parameters (trace(S)):":>42}  {self.tr_S:<.3f}\n'
    summary += f'{"Degree of freedom (n - trace(S)):":>42}  {self.df_model:<.3f}\n'
    summary += f'{"Sigma estimate:":>42}  {np.sqrt(self.sigma2):<.3f}\n'
    summary += f'{"Log-likelihood:":>42}  {self.llf:<.3f}\n'
    summary += f'{"AIC:":>42}  {self.aic:<.3f}\n'
    summary += f'{"AICc:":>42}  {self.aicc:<.3f}\n'
    summary += f'{"BIC:":>42}  {self.bic:<.3f}\n'
    summary += f'{"CV:":>42}  {self.cv:<.3f}\n'
    summary += f'{"R2:":>42}  {self.R2:<.3f}\n'
    summary += f'{"Adjusted R2:":>42}  {self.adj_R2:<.3f}\n\n'

    summary += f'{"Summary Statistics For MGTWR Parameter Estimates":^80}\n'
    summary += f'{"-":-^80}\n'

    summary += f'{"Variable":^15}{"Mean":^13}{"STD":^13}{"Min":^13}{"Median":^13}{"Max":^13}\n'
    summary += f'{"-":-<15} {"-":-^12} {"-":-^12} {"-":-^12} {"-":-^12} {"-":->12}\n'

    for i in range(self.k):
        summary += (f'{XNames[i]:^15}{np.mean(self.params[:, i]):^13.3f}{np.std(self.params[:, i]):^13.3f}'
                    f'{np.min(self.params[:, i]):^13.3f}{np.median(self.params[:, i]):^13.3f}'
                    f'{np.max(self.params[:, i]):^13.3f}\n')
    summary += f'{"=":=^80}\n'

    return summary
