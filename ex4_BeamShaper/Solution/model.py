import control
import matplotlib.pyplot as plt
import numpy as np

from inputs import disturbance

def open_loop_x():
    '''
    Model of the beam shaping in the x-direction. Inputs to the model in correct order: u_x, w_ex, w_ix. Outputs of the model: delta_x, f_x.

    Returns
    -------
        Px_T : State space model

    See: https://nl.mathworks.com/help/control/ug/thickness-control-for-a-steel-beam.html
    '''
    Hx = control.tf([2.4*10**8], [1, 72, 90**2], inputs='u_x', outputs='f1ax', name='Hx')
    Fex = control.tf([3*10**4, 0], [1, 0.125, 6**2], inputs='w_{ex}', outputs='f1bx', name='Fex')
    Fix = control.tf([10**4], [1, 0.05], inputs='w_{ix}', outputs='f2x')
    gx = 10**(-6)
    sum_blck = control.summing_junction(inputs=['f1ax', 'f1bx'], output='f1x', name='sumx')
    T1 = control.interconnect([Hx, Fex, sum_blck], inputs=['u_x', 'w_{ex}'], outputs='f1x')

    # T1.connection_table(show_names=True)
    T = control.append(T1, Fix)
    Px_matrix = np.array([[-gx, gx], [1, 1]])

    C_new = Px_matrix @ T.C
    D_new = Px_matrix @ T.D

    Px_T = control.ss(T.A, T.B, C_new, D_new)

    return Px_T


def open_loop_y():
    '''
    Model of the beam shaping in the y-direction. Inputs to the model in correct order: u_y, w_ey, w_iy. Outputs of the model: delta_y, f_y.

    Returns
    -------
        Py_T : State space model

    See: https://nl.mathworks.com/help/control/ug/thickness-control-for-a-steel-beam.html
    '''
    Hy = control.tf([7.8e8], [1, 71, 88**2], inputs='u_y', outputs='f1ay', name='Hx')
    Fey = control.tf([1e5, 0], [1, 0.19, 9.4**2], inputs='w_{ey}', outputs='f1by', name='Fey')
    Fiy = control.tf([2e4], [1, 0.05], inputs='w_{iy}', outputs='f2y')
    gy = 0.5e-6
    sum_blck = control.summing_junction(inputs=['f1ay', 'f1by'], output='f1y', name='sumy')

    T1 = control.interconnect([Hy, Fey, sum_blck], inputs=['u_y', 'w_{ey}'], output='f1y')
    T = control.append(T1, Fiy)
    # T1.connection_table(show_names=True)
    Py_matrix = np.array([[-gy, gy], [1, 1]])

    C_new = Py_matrix @ T.C
    D_new = Py_matrix @ T.D

    Py_T = control.ss(T.A, T.B, C_new, D_new)
    return Py_T


def cross_coupling():
    '''
    Model of the beam shaping mechanism in x- and y-direction. The interaction between x and y is incorporated.

    Input to the model in the correct order: u_x, w_ex, w_ix, u_y, w_ey, w_iy.
    Outputs of the model in the correct order: delta_x, f_x, delta_y, f_y

    Returns
    -------
        Pxy_T : State space model

    See: https://nl.mathworks.com/help/control/ug/thickness-control-for-a-steel-beam.html
    '''
    gxy = 0.1
    gyx = 0.4
    gx = 1e-6
    gy = 0.5e-6

    Px = open_loop_x()    
    Py = open_loop_y()

    Txy = control.append(Px,Py)
    CC = np.array([[1, 0, 0, gx*gyx], 
                   [0, 1, 0, -gyx],
                   [0, gy*gxy, 1, 0],
                   [0, -gxy, 0, 1]])
    Cxy_new = CC @ Txy.C
    Dxy_new = CC @ Txy.D

    Pxy_T = control.ss(Txy.A, Txy.B, Cxy_new, Dxy_new)
    return Pxy_T


def show_bode(model):
    '''
    Plot the Bode plot of the model.

    Inputs
    ------
        model: State space model
    '''
    omega = np.logspace(-2, 2, 1000)  
    mag, phase, omega = control.bode_plot(model, omega, dB=True, Hz=False, plot=True)
    plt.grid(True)
    plt.show()



def show_response():
    '''
    Create a time response of the coupled model for the disturbances.
    '''
    model = cross_coupling()
    t, u = disturbance()
    t, y = control.forced_response(model, T=t, U=u)

    plt.figure()
    plt.plot(t, y[0, :], label="delta_x")
    plt.plot(t, y[2, :], label='delta_y')
    plt.xlabel('t [s]')
    plt.ylabel('delta_i [m]')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Create the open-loop model in the x-direction and create a Bode.
    model_x = open_loop_x()
    show_bode(model_x)

    # Show the time response of the coupled model with disturbances.
    show_response()