import control
import matplotlib.pyplot as plt
import numpy as np

def step(x=False, y=False):
    ''' 
    Create step inputs for u_x and for u_y.
    If 'x' is True, only a step on u_x is returned. If 'y' is True only a step on u_y is returned. If both are False, both u_x and u_y are stepped.

    Returns
    -------
        t : time vector in seconds
        u : the input vector [u_x, w_ex, w_ix, u_y, w_ey, w_iy]
    '''
    dt = 0.0001
    t = np.arange(0, 0.3, dt)
    
    ux_size = 0 if not x else 10
    ux = ux_size * np.ones(len(t))
    wex = np.zeros(len(t))
    wix = np.zeros(len(t))
    uy_size = 0 if not y else 10
    uy = uy_size * np.ones(len(t))
    wey = np.zeros(len(t))
    wiy = np.zeros(len(t))

    u = np.array([ux, wex, wix, uy, wey, wiy])

    return t, u

def disturbance(x=False, y=False):
    ''' 
    Create disturbance inputs for w_i and for w_e.
    If 'x' is True, only a disturbance on w_ix and w_ex is returned. If 'y' is True only a disturbance on w_iy and w_ey is returned. If both are False, all disturbances are returned.

    Returns
    -------
        t : time vector in seconds
        u : the input vector [u_x, w_ex, w_ix, u_y, w_ey, w_iy]
    '''
    dt = 0.01
    t = np.arange(0, 30, dt)

    ux = np.zeros(len(t))
    wex = np.sqrt(1/dt) * np.random.normal(size=len(t))
    wix = np.sqrt(1/dt) * np.random.normal(size=len(t)) 
    uy = np.zeros(len(t))
    wey = np.sqrt(1/dt) * np.random.normal(size=len(t))
    wiy = np.sqrt(1/dt) * np.random.normal(size=len(t))
    
    if not x and not y:
        u = np.array([ux, wex, wix, uy, wey, wiy])
    elif x:
        u = np.array([uy, wey, wiy])
    elif y:
        u = np.array([ux, wex, wix])
    return t, u