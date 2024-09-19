import numpy as np
import control
import matplotlib.pyplot as plt

from LQRcontrol import LQR
from model import cross_coupling, open_loop_x, open_loop_y 
from inputs import disturbance, step

def MIMO_decoupler(model, tf_x, tf_y, plot=False):
    '''
    Implementation of a static decoupler between u_x and f_y on the one hand, and u_y and f_x on the other hand.

    Inputs
    ------
        model : State space model of the complete system.
        tf_x : State space model from u_x to delta_x.
        tf_y : State space model from u_y to delta_y.
        plot : Show the time responses of f_x and f_y in de coupled and decoupled open loop.

    Returns
    -------
        Gd : the decoupler matrix.
    '''
    G_full = control.dcgain(model)
    
    ###########################################################
    ####    Obtain the decoupling matrix Gd and implement #####
    ###########################################################
    # Obtain the decoupling matrix
    Gd = None

    # Create state space model of the process with decoupler (no closed-loop needed)
    B_new = None
    D_new = np.zeros((4, 2))
    model_decoupled = control.ss(model.A, B_new, model.C, D_new)

    # Time responses for steps
    t, u_x = step(x=True)
    t, u_y = step(y=True)
    t, y_x = control.forced_response(model, T=t, U=u_x)
    t, y_y = control.forced_response(model, T=t, U=u_y)

    t, yd_x = control.forced_response(model_decoupled, T=t, U=np.array([u_x[0, :], u_x[3, :]]))
    t, yd_y = control.forced_response(model_decoupled, T=t, U=np.array([u_y[0, :], u_y[3, :]]))

    if plot: 
        fig1 = plt.figure()
        plt.subplot(121)
        plt.plot(t, y_x[1, :], label='without decoupler')
        plt.plot(t, yd_x[1, :], label='with decoupler')
        plt.legend()
        plt.ylabel('$f_x$ [N]')
        plt.xlabel('t [s]')

        plt.subplot(122)
        plt.plot(t, y_x[3, :], label='without decoupler')
        plt.plot(t, yd_x[3, :], label='with decoupler')
        plt.ylabel('$f_y$ [N]')
        plt.legend()
        plt.xlabel('t [s]')
        fig1.suptitle('Step on $u_x$.')


        fig2 = plt.figure()
        plt.subplot(121)
        plt.plot(t, y_y[1, :], label='without decoupler')
        plt.plot(t, yd_y[1, :], label='with decoupler')
        plt.legend()
        plt.ylabel('$f_x$ [N]')
        plt.xlabel('t [s]')

        plt.subplot(122)
        plt.plot(t, y_y[3, :], label='without decoupler')
        plt.plot(t, yd_y[3, :], label='with decoupler')
        plt.legend()
        plt.ylabel('$f_y$ [N]')
        plt.xlabel('t [s]')
        plt.legend()
        fig2.suptitle('Step on $u_y$.')
        plt.show()
    return Gd

def MIMO_decentralised(Kx, Ky, model, tf_x, tf_y, Gd, plot=False):
    '''
    The LQR controller is designed for the x and y model, without taking the interactions into account. Then both controllers are applied to the full system. This strategy is called decentralised control. It is assumed that all states are measurable. If not, a Kalman filter should be added for each controller.

    Inputs
    -------
        Kx : LQR gains for the x model.
        Ky : LQR gains for the y model.
        model : State space model of the full system. 
        tf_x : State space model from u_x to delta_x.
        tf_y : State space model from u_y to delta_y.
        Gd : decoupling matrix.

    Returns
    --------
        t : time vector in seconds.
        y_cl : Closed-loop time response of disturbance.
    '''
    # Create closed-loop state-space model for decentralised control
    A_extra = np.block([[tf_x.B @ Kx, np.zeros_like(tf_x.A)],
                [np.zeros_like(tf_x.A), tf_y.B @ Ky]])
    A_n = model.A - A_extra
    cl = control.ss(A_n, model.B, model.C, model.D)

    # Add decoupler to the closed-loop response
    A_decoupled = np.block([[tf_x.B, np.zeros_like(tf_x.B)],
                            [np.zeros_like(tf_y.B), tf_y.B]]) @ Gd @ np.block([[Kx, np.zeros_like(Kx)], [np.zeros_like(Ky), Ky]])
    A_nd = model.A - A_decoupled
    cl_d = control.ss(A_nd, model.B, model.C, model.D)

    # Time responses
    t, u = disturbance()
    t, y = control.forced_response(model, T=t, U=u)
    t, y_cl = control.forced_response(cl, T=t, U=u)
    t, y_cld = control.forced_response(cl_d, T=t, U=u)

    if plot:
        fig = plt.figure()
        y_labels = ['$\delta_x$', '$f_x$', '$\delta_y$', '$f_y$']
        for i in range(len(y)):
            plt.subplot(1, len(y), i + 1)
            plt.plot(t, y[i, :], label='Open-loop')
            plt.plot(t, y_cld[i, :], label='Closed-loop with  decoupler')
            plt.plot(t, y_cl[i, :], label='Closed-loop')
            plt.ylabel(y_labels[i])
            plt.xlabel('t [s]')
            plt.legend()
        fig.suptitle('Decentralised LQR control.')
        plt.show()

    return t, y_cl

    
def MIMO_centralised(model, t, y_cl, plot=False):
    '''
    Centralised LQR control is used to take into account the interaction as well.  Now LQR gains are obtained for u_x and u_y. It is assumed that all states are measurable. If this is not the case, a Kalman filter can be designed to estimate the states.

    Inputs
    ------
        model : State space model of the full system.
        t : time vector in seconds.
        y_cl : time response of the decentralised LQR controllers.
        plot : Show the time responses due to disturbance.

    Returns
    -------
        Kxy : LQR gains
        tf : State space model from u_x and u_y to delta_x, delta_y
    '''
    # Create the model from u_x, u_y to delta_x, delta_y
    B_n = np.zeros((10, 2))
    B_n[:, 0] = model.B[:, 0] # u_x
    B_n[:, 1] = model.B[:, 3] # u_y
    C_n = np.zeros((2, 10))
    C_n[0, :] = model.C[0, :] # delta_x
    C_n[1, :] = model.C[2, :] # delta_y
    D_n = np.zeros((2, 2))
    tf = control.ss(model.A, B_n, C_n, D_n)

    # Obtain LQR gains
    Kxy, _, _ = control.lqr(tf.A, tf.B, tf.C.T @ tf.C, 1e-4*np.eye(2))

    # Create closed-loop model
    clxy = control.ss(model.A - tf.B @ Kxy, model.B, model.C, model.D)

    # Time responses
    t, u = disturbance()
    t, y = control.forced_response(model, T=t, U=u)
    t, y_clxy = control.forced_response(clxy, T=t, U=u)

    if plot:
        plt.figure()
        plt.subplot(121)
        plt.plot(t, y[0, :], label='Open-loop')
        plt.plot(t, y_clxy[0, :], label='Closed-loop centralised')
        plt.plot(t, y_cl[0, :], label='Closed-loop decentralised')
        plt.xlabel('t [s]')
        plt.ylabel('$\delta_x$ [m]')
        plt.legend()
        plt.subplot(122)
        plt.plot(t, y[2, :], label='Open-loop')
        plt.plot(t, y_clxy[2, :], label='Closed-loop centralised')
        plt.plot(t, y_cl[2, :], label='Closed-loop decentralised')
        plt.legend()
        plt.xlabel('t [s]')
        plt.ylabel('$\delta_y$ [m]')
        plt.show()
    return Kxy, tf


if __name__ == "__main__":
    # Create models
    model_x = open_loop_x()
    model_y = open_loop_y()
    model = cross_coupling()

    # Create decentralised LQR filters
    plot_filters = False
    Kx, tf_lqr_x = LQR(model_x, plot=plot_filters)
    Ky, tf_lqr_y = LQR(model_y, plot=plot_filters)

    # Create decoupler matrix
    Gd = MIMO_decoupler(model, tf_lqr_x, tf_lqr_y, plot=True)

    # Implement decentralised LQR control
    t, y_cl = MIMO_decentralised(Kx, Ky, model, tf_lqr_x, tf_lqr_y, Gd, plot=True)

    # Implement centralised LQR control
    MIMO_centralised(model, t, y_cl, plot=True)
    