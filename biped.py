import matplotlib.pyplot as plt
from numpy import *

__all__ = ['Biped']


def cpx2cart(y):
    """Complex array to cartesion coords: (real,imag)
    INPUTS:
        y -- number or array of numbers
    OUTPUTS:
        real(y), imag(y)
    """
    y = asarray(y)
    return real(y), imag(y)


def ang2cpx(a):
    """angle to complex phasor"""
    a = asarray(a, dtype=float)
    if a.ndim == 0:
        a = a[newaxis]
    a %= 2 * pi
    out = cos(a) + 1j * sin(a)
    zmsk = a == 0
    out[~zmsk] /= abs(out[~zmsk])
    return out


class Biped:
    """Planar biped dynamics/kinematics class
    Variable Definitions:
        x -- 6x1 -- state vector: (x, y, theta, xdot, ydot, omega)
        r -- 4x1 -- shape vector: (L1, L2, R1, R2)
    """

    def __init__(self, ell, m):
        self.ell = asarray(ell)
        self.m = asarray(m)
        self.basecfg = ell * asarray([-1j, -1j, -1j, -1j])

    def __call__(self, x, r, t):
        """biped dynamics
        INPUTS:
            x -- state vector
            r  -- shape vector
            t -- time value
        OUTPUTS:
            xdot -- 6x1
        """
        raise NotImplementedError

    def _find_pivot(self, x, r):
        """find foot on ground, if applicable
        INPUTS:
            x -- state vector
            r -- shape vector
            t -- float -- time value
        OUTPUTS:
            TODO
        """
        raise NotImplementedError

    def _2w(self, x, r):
        """get world-frame joint coords
        INPUTS:
            x -- state vector
            r -- shape vector
        OUTPUTS:
            array of joint xy coords: [body, L1, L2, R1, R2]
        """
        x, r = asarray(x), asarray(r)
        o = x[0] + 1j * x[1]
        oang = ang2cpx(x[2])
        c = ang2cpx(r)

        cfg = self.basecfg * asarray([c[0], c[0] * c[1], c[2], c[2] * c[3]])
        cfg *= oang
        wcfg = asarray([0, cfg[0], cfg[0] + cfg[1], cfg[2], cfg[2] + cfg[3]])
        wcfg += o
        return wcfg

    def plot(self, x, r, ax=None):
        if ax is None:
            ax = plt.figure().gca()
        wcfg = self._2w(x, r)
        ax.plot(*cpx2cart(wcfg[:3]), '.-')
        ax.plot(*cpx2cart(wcfg[[0, 3, 4]]), '.-')
        return ax


if __name__ == '__main__':
    print('Running unit tests...')
    print('Plotting kinematics...')
    b = Biped(1, 1)
    x = [1, 1, 0.5, 0, 0, 0]
    r = [-pi / 5, -0.4, 0.3, -0.15]
    f = plt.figure('biped-kin')
    f.clf()
    ax = f.add_subplot(111)
    b.plot(x, r, ax=ax)
    ax.grid()
    ax.set_aspect('equal')
    ttlstr = 'o=' + str(around(x[:3], 2)) + ';  ' + 'r=' + str(around(r, 2))
    ax.set_title(ttlstr)

    try:
        get_ipython()
        plt.ion()
    except NameError:
        plt.ioff()
        plt.show()
