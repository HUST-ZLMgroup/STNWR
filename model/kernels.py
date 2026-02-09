import numpy as np
from numba import njit

STA_W1 = np.loadtxt('data\\地理距离矩阵.csv', delimiter=',')
STA_W2 = np.loadtxt('data\\网络距离矩阵.csv', delimiter=',')
STA_W = STA_W1 + STA_W2


@njit
def local_cdist(coords_i, coords, spherical):
    """
    Compute Haversine (spherical=True) or Euclidean (spherical=False) distance for a local kernel.
    """
    if spherical:
        dLat = np.radians(coords[:, 1] - coords_i[1])
        dLon = np.radians(coords[:, 0] - coords_i[0])
        lat1 = np.radians(coords[:, 1])
        lat2 = np.radians(coords_i[1])
        a = np.sin(dLat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dLon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        R = 6371000.0

        return np.sqrt((R * c) ** 2 + (coords_i[2] - coords[:, 2]) ** 2)
    else:
        rst = np.sqrt(np.sum((coords_i - coords) ** 2, axis=1))
        if np.any(np.isnan(rst)):
            print(coords_i, coords, spherical)
            raise ValueError('0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000')
        return rst


class Kernel(object):
    """
    GWR kernel function specifications.
    """

    def __init__(self, i, coords, bw=None, fixed=True, kernel='triangular', spherical=False, points=None):
        '''
        if points is None:
            self.dvec = local_cdist(coords[i], coords, spherical).reshape(-1)
        else:
            self.dvec = local_cdist(points[i], coords, spherical).reshape(-1)
        '''
        self.dvec = STA_W[i]

        self.function = kernel

        if fixed:
            self.bandwidth = float(bw)
        else:
            self.bandwidth = np.partition(self.dvec, int(bw) - 1)[int(bw) - 1] * 1.0001  # partial sort in O(n) Time
        self.kernel = self._kernel_funcs(self.dvec / self.bandwidth)
        if self.function == "bisquare":  # Truncate for bisquare
            self.kernel[self.dvec >= self.bandwidth] = 0

    def _kernel_funcs(self, zs):
        # functions follow Anselin and Rey (2010) table 5.4
        if self.function == 'triangular':
            return 1 - zs
        elif self.function == 'uniform':
            return np.ones(zs.shape) * 0.5
        elif self.function == 'quadratic':
            return (3. / 4) * (1 - zs ** 2)
        elif self.function == 'quartic':
            return (15. / 16) * (1 - zs ** 2) ** 2
        elif self.function == 'gaussian':
            return np.exp(-0.5 * zs ** 2)
        elif self.function == 'bisquare':
            return (1 - zs ** 2) ** 2
        elif self.function == 'exponential':
            return np.exp(-zs)
        else:
            print('Unsupported kernel function', self.function)


if __name__ == '__main__':
    coords = np.array([[108.28, 31.64, 0],  # 154.356
                       [109.69, 32.34, 0],
                       [112.50, 33.05, 0]])  # 274.963

    print(local_cdist(coords[0], coords, True))
