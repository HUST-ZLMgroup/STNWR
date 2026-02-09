from .model import GTWR, One_Var_GTWR
from .search import golden_section, equal_interval, multi_bw
from .kernels import *
from .diagnostics import get_AICc, get_AIC, get_BIC, get_CV

getDiag = {'AICc': get_AICc, 'AIC': get_AIC, 'BIC': get_BIC, 'CV': get_CV}


class Sel_BW(object):
    def __init__(self, coords, y, X, mode='gtwr', kernel='bisquare', fixed=False, constant=True, spherical=False):
        if not coords.shape[0] == y.shape[0] and y.shape[0] == X.shape[0]:
            raise ValueError('Coords, X and y must have the same length.')
        if coords.shape[1] < 2 or coords.shape[1] > 3 or y.shape[1] > 1:
            raise ValueError('Errors in the shape of the data.')
        self.n = y.shape[0]  # 点的个数
        if coords.shape[1] == 2:
            self.coords = np.hstack((coords, np.zeros((self.n, 1))))
        else:
            self.coords = coords
        self.y = y.reshape(-1, 1)
        if constant:
            self.X = np.hstack((np.ones((self.n, 1)), X))
        else:
            self.X = X
        self.k = self.X.shape[1]  # 变量个数
        self.mode = mode.lower()
        self.kernel = kernel.lower()
        self.fixed = fixed
        self.spherical = spherical

    def search(self, search_method='golden_section', criterion='AICc',
               bw_min=None, bw_max=None, tau_min=0., tau_max=3., tau_decimal=5,
               bw_interval=1., tau_interval=0.1, tol=1.0e-6, max_iter=200, verbose=False,
               multi_init=None, tol_multi=1.0e-5, rss_score=False, max_iter_multi=200, multi_bw_min=None,
               multi_bw_max=None, max_same_times=5, pool=None):
        """
        :param search_method: golden_section or equal_interval
        :param criterion: AICc AIC BIC CV
        :param bw_min:
        :param bw_max:
        :param tau_min:
        :param tau_max:
        :param tau_decimal:
        :param bw_interval: 当search_method为equal_interval时使用
        :param tau_interval: 当search_method为equal_interval时使用
        :param tol:
        :param max_iter:
        :param verbose:
        :param multi_init: [bw, tau, resid_response, params]
        :param tol_multi:
        :param rss_score:
        :param max_iter_multi:
        :param multi_bw_min:
        :param multi_bw_max:
        :param max_same_times:
        :param pool:
        :return:
        """
        self.search_method = search_method
        self.criterion = criterion
        self.bw_min = bw_min
        self.bw_max = bw_max
        if self.mode == 'gwr' or self.mode == 'mgwr':
            self.temporally = False
            self.tau_min = 0.
            self.tau_max = 0.
        elif self.mode == 'gtwr' or self.mode == 'mgtwr':
            self.temporally = True
            self.tau_min = tau_min
            self.tau_max = tau_max
        else:
            raise AttributeError(f'Unsupported mode {self.mode}.')
        self.tau_decimal = tau_decimal
        self.bw_interval = bw_interval
        self.tau_interval = tau_interval
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.int_score = not self.fixed
        self.pool = pool

        if self.mode == 'gwr' or self.mode == 'gtwr':
            self._bw_tau()
            self.pool = None
            return self.search_result[0]
        else:
            if multi_init is None:
                self._bw_tau()
                bw, tau = self.search_result[0]
                gtwr = GTWR(self.coords, self.y, self.X, bw, tau, temporally=self.temporally, kernel=self.kernel,
                            fixed=self.fixed, constant=False, spherical=self.spherical).fit(lite=True, pool=self.pool)
                resid_response = gtwr.resid_response
                params = gtwr.params
                self.multi_init = [bw, tau, resid_response, params]
            else:
                self.multi_init = multi_init
            self.tol_multi = tol_multi
            self.rss_score = rss_score
            self.max_iter_multi = max_iter_multi

            if multi_bw_min is None:
                if self.fixed:
                    self.multi_bw_min = [min([np.min(np.delete(
                        local_cdist(self.coords[i], self.coords, spherical=self.spherical), i))
                        for i in range(self.n)])] * self.k
                else:
                    self.multi_bw_min = [2.] * self.k
            elif isinstance(multi_bw_min, (int, float)):
                self.multi_bw_min = [multi_bw_min] * self.k
            elif len(multi_bw_min) == self.k:
                self.multi_bw_min = multi_bw_min
            else:
                raise AttributeError(
                    "multi_bw_min must be either a list containing"
                    " a single entry or a list containing an entry for each of k"
                    " covariates including the intercept")

            if multi_bw_max is None:
                if self.fixed:
                    self.multi_bw_max = [max([np.max(
                        local_cdist(self.coords[i], self.coords, spherical=self.spherical))
                        for i in range(self.n)])] * self.k
                else:
                    self.multi_bw_max = [self.n] * self.k
            elif isinstance(multi_bw_max, (int, float)):
                self.multi_bw_max = [multi_bw_max] * self.k
            elif len(multi_bw_max) == self.k:
                self.multi_bw_max = multi_bw_max
            else:
                raise AttributeError(
                    "multi_bw_max must be either a list containing"
                    " a single entry or a list containing an entry for each of k"
                    " covariates including the intercept")

            self.max_same_times = max_same_times
            self._mbw_mtau()
            self.pool = None
            return self.search_result_multi[0]

    def _bw_tau(self):
        def gtwr_func(bw, tau):
            return getDiag[self.criterion](GTWR(
                self.coords, self.y, self.X, bw, tau, temporally=self.temporally, kernel=self.kernel, fixed=self.fixed,
                constant=False, spherical=self.spherical).fit(lite=True, pool=self.pool))[0]  # + np.sqrt(tau)

        if self.search_method == 'golden_section':
            a, c = self._init_section()
            delta = 0.38197  # 1 - (np.sqrt(5.0)-1.0)/2.0
            self.search_result = golden_section(a, c, self.tau_min, self.tau_max, self.tau_decimal, delta, gtwr_func,
                                                self.tol, self.max_iter, self.int_score, self.verbose)
        elif self.search_method == 'equal_interval':
            self.search_result = equal_interval(self.bw_min, self.bw_max, self.bw_interval,
                                                self.tau_min, self.tau_max, self.tau_interval,
                                                gtwr_func, self.int_score, self.verbose)
        else:
            raise TypeError(f'Unsupported computational search method {self.search_method},'
                            ' only "golden_section" and "equal_interval" is supported')

    def _mbw_mtau(self):

        def gtwr_func(y, X, bw, tau):
            return One_Var_GTWR(self.coords, y, X, bw, tau, kernel=self.kernel,
                                fixed=self.fixed, spherical=self.spherical).fit(pool=self.pool)

        def sel_func(y, X, bw_min, bw_max):
            return _One_Var_Sel(self.coords, y, X, kernel=self.kernel,
                                fixed=self.fixed, spherical=self.spherical).search(
                search_method=self.search_method, criterion=self.criterion,
                bw_min=bw_min, bw_max=bw_max, tau_min=self.tau_min, tau_max=self.tau_max, tau_decimal=self.tau_decimal,
                bw_interval=self.bw_interval, tau_interval=self.tau_interval, tol=self.tol, max_iter=self.max_iter,
                pool=self.pool)

        self.search_result_multi = multi_bw(self.multi_init, self.y, self.X, self.n, self.k, self.tol_multi,
                                            self.max_iter_multi, self.rss_score, gtwr_func, sel_func,
                                            self.multi_bw_min, self.multi_bw_max, self.max_same_times,
                                            verbose=self.verbose)

    def _init_section(self):
        n_vars = self.k
        n = self.n

        if self.int_score:
            a = 40 + 2 * n_vars
            c = n
        else:
            min_dist = np.min(np.array([np.min(np.delete(
                local_cdist(self.coords[i], self.coords, spherical=self.spherical), i))
                for i in range(n)]))
            max_dist = np.max(np.array([np.max(
                local_cdist(self.coords[i], self.coords, spherical=self.spherical))
                for i in range(n)]))

            a = min_dist / 2.0
            c = max_dist * 2.0

        if self.bw_min is not None:
            a = self.bw_min
        if self.bw_max is not None and self.bw_max is not np.inf:
            c = self.bw_max

        return a, c


class _One_Var_Sel(object):
    def __init__(self, coords, y, X, kernel='bisquare', fixed=False, spherical=False):
        self.n = y.shape[0]  # 点的个数
        self.coords = coords
        self.y = y
        self.X = X
        self.kernel = kernel.lower()
        self.fixed = fixed
        self.spherical = spherical

    def search(self, search_method='golden_section', criterion='AICc',
               bw_min=None, bw_max=None, tau_min=0., tau_max=3., tau_decimal=5,
               bw_interval=1., tau_interval=0.1, tol=1.0e-6, max_iter=200,
               pool=None):
        self.search_method = search_method
        self.criterion = criterion
        self.bw_min = bw_min
        self.bw_max = bw_max
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.tau_decimal = tau_decimal
        self.bw_interval = bw_interval
        self.tau_interval = tau_interval
        self.tol = tol
        self.max_iter = max_iter
        self.int_score = not self.fixed
        self.pool = pool

        self._bw_tau()
        self.pool = None
        return self.search_result[0]

    def _bw_tau(self):
        def gtwr_func(bw, tau):
            return getDiag[self.criterion](One_Var_GTWR(
                self.coords, self.y, self.X, bw, tau, kernel=self.kernel, fixed=self.fixed,
                spherical=self.spherical).fit(pool=self.pool))  # + np.sqrt(tau)

        if self.search_method == 'golden_section':
            delta = 0.38197  # 1 - (np.sqrt(5.0)-1.0)/2.0
            self.search_result = golden_section(self.bw_min, self.bw_max, self.tau_min, self.tau_max, self.tau_decimal, delta, gtwr_func,
                                                self.tol, self.max_iter, self.int_score)
        elif self.search_method == 'equal_interval':
            self.search_result = equal_interval(self.bw_min, self.bw_max, self.bw_interval,
                                                self.tau_min, self.tau_max, self.tau_interval,
                                                gtwr_func, self.int_score)
        else:
            raise TypeError(f'Unsupported computational search method {self.search_method},'
                            ' only "golden_section" and "equal_interval" is supported')
