from scipy import linalg
from tqdm import tqdm
from .kernels import *
from .results import GTWRResults, GTWRResultsLite, MGTWRResults


def Model(coords, y, X, bw=None, tau=None, selector=None, mode='gtwr', kernel='bisquare',
          fixed=False, constant=True, spherical=False):
    """
    :param coords:
    :param y:
    :param X:
    :param bw: Single scale parameter, only valid if mode=gwr or gtwr.
    :param tau: Single scale parameter, only valid if mode=gwr.
    :param selector: Multiscale parameter, valid only if mode=mgwr or mgtwr.
    :param mode: The fitting method, which can only be gwr, gtwr, mgwr or mgtwr.
    :param kernel:
    :param fixed:
    :param constant:
    :param spherical:
    """
    mode = mode.lower()
    if mode == 'gwr' or mode == 'gtwr':
        temporally = True if (mode == 'gtwr' or mode == 'mgtwr') else False
        return GTWR(coords=coords, y=y, X=X, bw=bw, tau=tau, temporally=temporally, kernel=kernel, fixed=fixed,
                    constant=constant, spherical=spherical)
    elif mode == 'mgwr' or mode == 'mgtwr':
        return MGTWR(coords=coords, y=y, X=X, selector=selector, kernel=kernel, fixed=fixed,
                     constant=constant, spherical=spherical)
    else:
        raise AttributeError(f'Unsupported mode {mode}.')


class GTWR(object):
    def __init__(self, coords, y, X, bw, tau=0., temporally=True, kernel='bisquare', fixed=False, constant=True,
                 spherical=False):
        if not coords.shape[0] == y.shape[0] and y.shape[0] == X.shape[0]:
            raise ValueError('coords, X and y must have the same length.')
        if coords.shape[1] < 2 or coords.shape[1] > 3 or y.shape[1] > 1:
            raise ValueError('Errors in the shape of the data.')
        self.n = y.shape[0]  # 点的个数
        self.bw = bw
        self.tau = tau if temporally else 0.
        if coords.shape[1] == 2:
            _t = np.zeros((self.n, 1))
        else:
            _t = np.sqrt(self.tau) * coords[:, 2].reshape((-1, 1))
        self.coords = np.hstack((coords[:, :2], _t))
        self.y = y.reshape(-1, 1)
        self.constant = constant
        if constant:
            self.X = np.hstack((np.ones((self.n, 1)), X))
        else:
            self.X = X
        self.k = self.X.shape[1]  # 变量个数
        self.kernel = kernel.lower()
        self.fixed = fixed
        self.spherical = spherical

    def _build_wi(self, i, bw):
        if bw == np.inf:
            wi = np.ones(self.n)
            return wi
        try:
            wi = Kernel(i, self.coords, bw,
                        fixed=self.fixed,
                        kernel=self.kernel,
                        spherical=self.spherical).kernel
        except Exception:
            raise
        return wi

    def _local_fit(self, i):
        wi = self._build_wi(i, self.bw).reshape(-1, 1)  # local spatial weights
        xTw = (self.X * wi).T
        xTwx = np.dot(xTw, self.X)
        xTwx_inv_xTw = linalg.solve(xTwx, xTw)
        betas = np.dot(xTwx_inv_xTw, self.y)

        predy = np.dot(self.X[i], betas)[0]
        resid = self.y[i, 0] - predy
        influ = np.dot(self.X[i], xTwx_inv_xTw[:, i])

        CCT = np.diag(np.dot(xTwx_inv_xTw, xTwx_inv_xTw.T)).reshape(-1)
        return influ, resid, predy, betas.reshape(-1), CCT

    def fit(self, lite=False, pool=None):
        if pool:
            rslt = pool.map(self._local_fit, range(self.n))
        else:
            rslt = map(self._local_fit, range(self.n))
        rslt_list = list(zip(*rslt))
        influ = np.array(rslt_list[0]).reshape(-1, 1)
        resid = np.array(rslt_list[1]).reshape(-1, 1)
        params = np.array(rslt_list[3])
        if lite:
            return GTWRResultsLite(self, resid, influ, params)
        else:
            predy = np.array(rslt_list[2]).reshape(-1, 1)
            CCT = np.array(rslt_list[4])
            return GTWRResults(self, params, predy, influ, CCT)

    def predict(self, coords, X, bw=None, tau=None):
        p_coords = coords.copy()
        p_X = X.copy()
        if bw is None:
            bw = self.bw
        if tau is None:
            tau = self.tau
        n = p_coords.shape[0]
        if self.constant:
            p_X = np.hstack((np.ones((n, 1)), p_X))
        p_coords[:, 2] = p_coords[:, 2] * np.sqrt(tau)
        prediction = []
        params = []
        for i in tqdm(range(n), desc='Predict'):
            wi = Kernel(i, self.coords, bw,
                        fixed=self.fixed,
                        kernel=self.kernel,
                        spherical=self.spherical,
                        points=p_coords).kernel.reshape(-1, 1)
            xTw = (self.X * wi).T
            xTwx = np.dot(xTw, self.X)
            xTwx_inv_xTw = linalg.solve(xTwx, xTw)
            betas = np.dot(xTwx_inv_xTw, self.y)

            predy = np.dot(p_X[i], betas)[0]
            prediction.append(predy)
            params.append(betas.reshape(-1))
        return np.array(prediction).reshape(-1, 1), np.array(params)


class One_Var_GTWR(object):
    def __init__(self, coords, y, X, bw, tau=0., kernel='bisquare', fixed=False, spherical=False):
        self.n = y.shape[0]
        _t = np.sqrt(tau) * coords[:, 2].reshape((-1, 1))
        self.coords = np.hstack((coords[:, :2], _t))
        self.y = y
        self.X = X
        self.bw = bw
        self.tau = tau
        self.kernel = kernel.lower()
        self.fixed = fixed
        self.spherical = spherical

    def _local_fit(self, i):
        wi = Kernel(i, self.coords, self.bw,
                    fixed=self.fixed,
                    kernel=self.kernel,
                    spherical=self.spherical).kernel
        xTw = self.X * wi  # (n)
        xTwx = np.sum(xTw * self.X)  # (1)
        xTwx_inv_xTw = xTw / xTwx  # (n)
        betas = np.sum(xTwx_inv_xTw * self.y)  # (1)
        predy = self.X[i] * betas
        resid = self.y[i] - predy
        influ = self.X[i] * xTwx_inv_xTw[i]
        return influ, resid, predy, betas

    def fit(self, pool=None):
        if pool:
            rslt = pool.map(self._local_fit, range(self.n))
        else:
            rslt = map(self._local_fit, range(self.n))
        rslt_list = list(zip(*rslt))
        influ = np.array(rslt_list[0])
        resid = np.array(rslt_list[1])
        params = np.array(rslt_list[3])
        return GTWRResultsLite(self, resid, influ, params)


class MGTWR(object):
    def __init__(self, coords, y, X, selector, kernel='bisquare', fixed=False, constant=True, spherical=False):

        if not coords.shape[0] == y.shape[0] and y.shape[0] == X.shape[0]:
            raise ValueError('coords, X and y must have the same length.')
        if coords.shape[1] < 2 or coords.shape[1] > 3 or y.shape[1] > 1:
            raise ValueError('Errors in the shape of the data.')
        self.selector = selector
        self.bws_taus = selector.search_result_multi[0]  # final set of bandwidth and tau
        self.bws_taus_history = selector.search_result_multi[1]  # bws and taus history in backfitting
        self.n = y.shape[0]  # 点的个数
        if coords.shape[1] == 2:
            self.coords = np.hstack((coords[:, :2], np.zeros((self.n, 1))))
        else:
            self.coords = coords
        self.y = y.reshape(-1, 1)
        self.constant = constant
        if constant:
            self.X = np.hstack((np.ones((self.n, 1)), X))
        else:
            self.X = X
        self.k = self.X.shape[1]  # 变量个数
        self.bw_init = selector.multi_init[0]
        self.tau_init = selector.multi_init[1]
        self.kernel = kernel.lower()
        self.fixed = fixed
        self.spherical = spherical

    def _build_wi(self, i, bw, tau, points=None):
        coords = np.hstack((self.coords[:, :2], np.sqrt(tau) * self.coords[:, 2].reshape((-1, 1))))
        wi = Kernel(i, coords, bw,
                    fixed=self.fixed,
                    kernel=self.kernel,
                    spherical=self.spherical,
                    points=points).kernel
        return wi

    def _chunk_compute_R(self, chunk_id=0):
        n = self.n
        k = self.k
        n_chunks = self.n_chunks
        chunk_size = int(np.ceil(float(n / n_chunks)))
        ENP_j = np.zeros(self.k)
        CCT = np.zeros((self.n, self.k))

        chunk_index = np.arange(n)[chunk_id * chunk_size:(chunk_id + 1) * chunk_size]
        init_pR = np.zeros((n, len(chunk_index)))
        init_pR[chunk_index, :] = np.eye(len(chunk_index))
        pR = np.zeros((n, len(chunk_index), k))  # partial R: n by chunk_size by k

        for i in range(n):
            wi = self._build_wi(i, self.bw_init, self.tau_init).reshape(-1, 1)
            xT = (self.X * wi).T
            P = np.linalg.solve(xT.dot(self.X), xT).dot(init_pR).T
            pR[i, :, :] = P * self.X[i]

        err = init_pR - np.sum(pR, axis=2)  # n by chunk_size

        for iter_i in range(self.bws_taus_history.shape[0]):
            for j in range(k):
                pRj_old = pR[:, :, j] + err
                Xj = self.X[:, j]
                n_chunks_Aj = n_chunks
                chunk_size_Aj = int(np.ceil(float(n / n_chunks_Aj)))
                for chunk_Aj in range(n_chunks_Aj):
                    chunk_index_Aj = np.arange(n)[chunk_Aj * chunk_size_Aj:(chunk_Aj + 1) * chunk_size_Aj]
                    pAj = np.empty((len(chunk_index_Aj), n))
                    for i in range(len(chunk_index_Aj)):
                        index = chunk_index_Aj[i]
                        wi = self._build_wi(index, self.bws_taus_history[iter_i, j, 0],
                                            self.bws_taus_history[iter_i, j, 1])
                        xw = Xj * wi
                        pAj[i, :] = Xj[index] / np.sum(xw * Xj) * xw
                    pR[chunk_index_Aj, :, j] = pAj.dot(pRj_old)
                err = pRj_old - pR[:, :, j]

        for j in range(k):
            CCT[:, j] += ((pR[:, :, j] / self.X[:, j].reshape(-1, 1)) ** 2).sum(axis=1)
        for i in range(len(chunk_index)):
            ENP_j += pR[chunk_index[i], i, :]

        return ENP_j, CCT

    def fit(self, n_chunks=1, pool=None):
        params = self.selector.search_result_multi[-2]
        predy = np.sum(self.X * params, axis=1).reshape(-1, 1)

        if pool:
            self.n_chunks = pool._processes * n_chunks
            rslt = tqdm(pool.imap(self._chunk_compute_R, range(self.n_chunks)),
                        total=self.n_chunks, desc='Inference')
        else:
            self.n_chunks = n_chunks
            rslt = map(self._chunk_compute_R, tqdm(range(self.n_chunks), desc='Inference'))

        rslt_list = list(zip(*rslt))
        ENP_j = np.sum(np.array(rslt_list[0]), axis=0)
        CCT = np.sum(np.array(rslt_list[1]), axis=0)

        return MGTWRResults(self, params, predy, CCT, ENP_j)

    def cal_Si(self, X, i, bw, tau):
        wi = self._build_wi(i, bw, tau)
        xTw = X * wi  # (n)
        xTwx = np.sum(xTw * X)  # (1)
        xTwx_inv_xTw = xTw / xTwx  # (n)
        Si = X[i] * xTwx_inv_xTw
        return Si

    def exact_fit(self):
        P = []
        Q = []
        I = np.eye(self.n)
        for j1 in range(self.k):
            Aj = np.array([self.cal_Si(self.X[:, j1], i, self.bws_taus[j1, 0], self.bws_taus[j1, 1])
                           for i in range(self.n)])
            Pj = []
            for j2 in range(self.k):
                if j1 == j2:
                    Pj.append(I)
                else:
                    Pj.append(Aj)
            P.append(Pj)
            Q.append([Aj])

        P = np.block(P)
        Q = np.block(Q)
        R = np.linalg.solve(P, Q)
        self.f = R.dot(self.y)  # 用于预测------------------------------------------------------------------------

        params = self.f / self.X.T.reshape(-1, 1)
        params = params.reshape(-1, self.n).T

        R = np.stack(np.split(R, self.k), axis=2)
        ENP_j = np.trace(R, axis1=0, axis2=1)
        predy = np.sum(self.X * params, axis=1).reshape(-1, 1)

        CCT = np.zeros((self.n, self.k))
        for j in range(self.k):
            CCT[:, j] = ((R[:, :, j] / self.X[:, j].reshape(-1, 1)) ** 2).sum(axis=1)
        self.err = self.y - predy  # 用于预测---------------------------------------------------------------------
        return MGTWRResults(self, params, predy, CCT, ENP_j)

    def predict(self, coords, X, bws_taus=None):
        try:
            _ = self.f
        except Exception:
            self.exact_fit()
        p_coords = coords.copy()
        p_X = X.copy()
        if bws_taus is None:
            bws_taus = self.bws_taus
        n = p_coords.shape[0]
        if self.constant:
            p_X = np.hstack((np.ones((n, 1)), p_X))
        params = np.zeros((n, self.k))
        ys = self.f.reshape(-1, self.n) + self.err.flatten()
        for i in tqdm(range(n), desc='Predict'):
            for j in range(self.k):
                cor = p_coords.copy()
                cor[:, 2] = cor[:, 2] * np.sqrt(bws_taus[j, 1])
                wi = self._build_wi(i, bws_taus[j, 0], bws_taus[j, 1], points=cor)

                xTw = self.X[:, j] * wi  # (n)
                xTwx = np.sum(xTw * self.X[:, j])  # (1)
                xTwx_inv_xTw = xTw / xTwx  # (n)
                params[i, j] = np.sum(xTwx_inv_xTw * ys[j])  # (1)

        return np.sum(p_X * params, axis=1).reshape(-1, 1), params
