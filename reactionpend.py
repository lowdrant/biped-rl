import sympy as sm
from numpy import *
from sympy import Matrix, expand, simplify, trigsimp


def get_dynam():
    """symbolically compute dynamics eqns"""

    t, r, L, m, g = sm.symbols('t r L m g')

    # Coordinates
    phi = sm.Function('phi')
    phi = phi(t)
    theta = sm.Function('theta')
    theta = theta(t)
    q = Matrix([phi, theta])
    qdot = q.diff(t)
    qddot = qdot.diff(t)

    # Kinematics
    x1 = r * sm.sin(phi) + L * sm.cos(phi - theta)
    x2 = r * sm.sin(phi) - L * sm.cos(phi - theta)
    y1 = r * sm.cos(phi) - L * sm.sin(phi - theta)
    y2 = r * sm.cos(phi) + L * sm.sin(phi - theta)
    p = Matrix([x1, y1, x2, y2])
    J = p.diff(q.T).reshape(2, 4).tomatrix().T
    v = J * qdot

    # Energy
    KE = m / 2 * v.T * v / 2
    KE = trigsimp(expand(KE))[0]  # 1x1 matrix to scalar
    PE = m / 2 * g * (y1 + y2)
    Lag = simplify(expand(KE - PE))
    Lag.diff(q)
    simplify(Lag.diff(qdot).diff(t))

    # Dynamics Matrices
    D = m / 2 * trigsimp(expand(J.T * J))
    G = PE.diff(q)
    Ccor = sm.diff(D * qdot, q.T).reshape(2, 2).tomatrix().T * qdot
    Cfug = sm.diff(qdot.T * D * qdot, q.T).reshape(1, 2).tomatrix().T / 2
    C = Ccor + Cfug

    # Control
    u = sm.symbols('u')
    B = Matrix([0, 1 / (m * L**2)])
    Dinv = simplify(D.inv())
    f = simplify(Dinv * B * u - Dinv * G)
    f = f.subs({phi: 'phi', theta: 'theta'})  # replace funcs with vars
    return f


def dynam2num(f, g, m, r, L):
    """convert sym dynamics to numeric eqn"""
    phi, theta, u = sm.symbols('phi theta u')
    fsubs = f.subs({'g': g, 'r': r, 'L': L, 'm': m})
    return sm.lambdify([phi, theta, u], fsubs, 'numpy')


def q2c(phi, theta, r, L):
    """coords to xy
    `plt.plot(*q2c(args))` should work out of the box
    OUT:
        (x,y)
        x := (xorigin, xpivot, x1, xpivot, x2)
    """
    x1 = r * sin(phi) + L * cos(phi - theta)
    x2 = r * sin(phi) - L * cos(phi - theta)
    y1 = r * cos(phi) - L * sin(phi - theta)
    y2 = r * cos(phi) + L * sin(phi - theta)
    xp = (x1 + x2) / 2
    yp = (y1 + y2) / 2
    x = (zeros_like(xp), xp, x1, xp, x2)
    y = (zeros_like(xp), yp, y1, yp, y2)
    return x, y


if __name__ == '__main__':
    try:
        faccel
    except NameError:
        fsym = get_dynam()
        g, r, L, m = 10, 1, 0.1, 1
        faccel = dynam2num(fsym, g, m, r, L)

    from numpy import *
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt

    def f(t, x, u=0):
        xdot = zeros_like(x)
        xdot[:2] = x[2:]
        # print(x)
        xdot[2:] = faccel(*x[:2], u).squeeze()
        # print(xdot)
        return xdot

    def ctl(t,x,r=0):
        err = r - x[0]
        errdot = r - x[2]
        u = 10*err + errdot
        return f(t,x,u)


    tf = 1
    t = linspace(0, tf, 1000*tf)
    x0 = [2, 0, 0.1, 0.]
    y = odeint(ctl, x0, t, tfirst=1, args=(0.0,))
    xy = asarray(q2c(*y[:, :2].T, r, L))

    f = plt.figure(1)
    f.clf()
    ax1 = f.add_subplot(231)
    ax2 = f.add_subplot(234, sharex=ax1)
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2, colspan=2, fig=f)
    ax1.plot(t, y[:, 0], '-', label='$\\phi$')
    ax1.plot(t, y[:, 2], '-', label='$\\dot{\\phi}$')
    ax2.plot(t,y[:,0]-y[:,1],'-',label='$\\phi-\\theta$')
    ax2.plot(t, y[:, 1], '-', label='$\\theta$')
    ax2.plot(t, y[:, 3], '-', label='$\\dot{\\theta}$')
    ax3.plot(*xy[:, 2], '-', label='mass 1', c='tab:orange')
    ax3.plot(*xy[:, -1], '-', label='mass 2', c='tab:green')
    ax3.plot(*xy[:, 1], '-', label='pivot', c='tab:blue', lw=3)
    ax3.plot(*xy[:,1,[0,-1]],'X',c='tab:blue',ms=10)
    ax3.plot(*xy[:,2,[0,-1]],'X',c='tab:orange',ms=10)
    ax3.plot(*xy[:,-1,[0,-1]],'X',c='tab:green',ms=10)
    ax3.plot(0, 0, 'k.')
    for ax in f.axes:
        ax.grid()
        ax.legend()
    # ax1.set_ylabel('angle [rad]')
    # ax2.set_ylabel('omega [rad/sec]')
    ax2.set_xlabel('t [sec]')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_aspect('equal')
    f.suptitle('PD controller')

    try:
        get_ipython()
        plt.ion()
    except NameError:
        plt.ioff()
        plt.show()
