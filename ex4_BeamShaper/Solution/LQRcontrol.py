import control
from control.matlab import lqr
import numpy as np
import matplotlib.pyplot as plt

from model import open_loop_x, open_loop_y
from inputs import disturbance


def LQR(model, x=True, y=False, plot=False):
    '''
    Apply LQR control to the x or y model.

    Inputs
    ------
        model : x or y model in state space form.
        x : the x model is used, if True. Should be False if y is True.
        y : the y model is used, if True. Should be False if x is True.
        plot : plot the time responses to compare open- en closed-loop.

    Returns
    -------
        Ki : the LQR gains.
        tf_i : state space model from u_i to delta_i.

    '''
    tf = model[0, 0] # From u_x to delta_x
    Ki, _, _ = lqr(tf.A, tf.B, tf.C.T @ tf.C, 1e-4)

    # Closed-loop state space model
    cli = control.ss(model.A - tf.B @ Ki, model.B, model.C, model.D)
    
    # Time responses
    t, u = disturbance(x=x, y=y)
    t, y = control.forced_response(model, T=t, U=u)
    t, y_cli = control.forced_response(cli, T=t, U=u)

    if plot:
        plt.figure()
        name = 'x' if x else 'y'
        plt.title("LQR control on the " + name + " model.")
        plt.plot(t, y[0, :], label='open-loop')
        plt.plot(t, y_cli[0, :], label='closed-loop')
        plt.xlabel('t [s]')
        plt.ylabel('delta_i [m]')
        plt.legend()
        plt.show()
    return Ki, tf

def kalman(model, plot=False):
    '''
    The Kalman filter is able to optimally estimate the states of a process, even with noise. This is shown for a time response using disturbance noise.

    Inputs
    ------
        model : State space model of the x or y model.
        plot : Show time response with estimate of the states.

    Returns
    -------
        Ki : the Kalman filter gains.
        tf_i : state space model from w_ei and w_ii to f_i

    '''
    tf = model[1, 1:] # from w_e and w_i to f
    Ex, _, _ = control.lqe(tf.A, tf.B, tf.C, np.eye(2), 1e4)

    # Closed-loop state space model
    A_kal = np.block([[model.A, np.zeros_like(model.A)],
                    [Ex @ tf.C, model.A - Ex @ tf.C]])
    B_kal = np.block([[tf.B], [np.zeros_like(tf.B)]])
    C_kal = np.block([[model.C, np.zeros_like(model.C)],
                    [np.zeros_like(model.C), model.C]])
    D_kal = np.zeros((len(C_kal), len(B_kal[0])))
    filt = control.ss(A_kal, B_kal, C_kal, D_kal)

    # Time responses
    t, u = disturbance(x=True)
    t, y_filt, x_filt = control.forced_response(filt, T=t, U=u[1:, :], return_x=True)

    if plot:
        n_st = int(len(x_filt)/2)
        fig = plt.figure()
        for i in range(n_st):
            plt.subplot(1, n_st, i + 1)
            plt.ylabel("State " + str(i + 1))
            plt.plot(t, x_filt[i, :], label='x(t)')
            plt.plot(t, x_filt[i + n_st, :], label='$\hat{x}(t)$')
            plt.xlabel('t [s]')
            plt.legend()
        fig.suptitle('Demo Kalman filter.')
        plt.show()

    return Ex, tf


def LQG(model, plot=False):
    '''
    The LQR controller needs all states, but often they cannot be measured. The Kalman filter is used to estimate the states, based on input and output measurements, which is then fed to the LQR controller. 

    Both the Bode and time responses are checked.

    Inputs
    -------
        model : State space model of the x or y model.
        plot : plot the Bode and time response due to disturbance.
    '''
    # Obtain LQR and Kalman Filter gains
    Kx, tf_lqr = LQR(model)
    Ex, tf_kal = kalman(model)
    
    # Closed-loop state space model
    A_lqg = np.block([[model.A, -tf_lqr.B @ Kx],
                    [Ex @ tf_kal.C, model.A - Ex @ tf_kal.C - tf_lqr.B @ Kx]])
    B_lqg = np.block([[tf_kal.B], [np.zeros_like(tf_kal.B)]])
    C_lqg = np.block([[model.C, np.zeros_like(model.C)],
                    [np.zeros_like(model.C), model.C]])
    D_lqg = np.zeros((len(C_lqg), len(B_lqg[0])))
    clx = control.ss(A_lqg, B_lqg, C_lqg, D_lqg)

    if plot:
        # Create Bode plots 
        Px_12 = model[0, 1] # from input w_ex to output delta_x
        Px_13 = model[0, 2] # from input w_ix to output delta_x
        clx_12 = clx[0, 0] # from input w_ex to output delta_x
        clx_13 = clx[0, 1] # from input w_ix to output delta_x

        omega = np.logspace(-1, 2, 500)
        mag_Px_12, phase, omega = control.bode(Px_12, omega, plot=False)
        mag_Px_13, phase, omega = control.bode(Px_13, omega, plot=False)
        mag_clx_12, phase, omega = control.bode(clx_12, omega, plot=False)
        mag_clx_13, phase, omega = control.bode(clx_13, omega, plot=False)

        fig = plt.figure()
        plt.subplot(121)
        plt.title('$w_{ex}$ to $\delta_x$')
        plt.loglog(omega, mag_Px_12, 'b', label='Open-loop')
        plt.loglog(omega, mag_clx_12, 'r', label='Closed-loop')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plt.xlabel('Frequency [rad/s]')
        plt.ylabel('Magnitude')

        plt.subplot(122)
        plt.title('$w_{ix}$ to $\delta_x$')
        plt.loglog(omega, mag_Px_13, 'b', label='Open-loop')
        plt.loglog(omega, mag_clx_13, 'r', label='Closed-loop')
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.ylabel('Magnitude')
        plt.xlabel('Frequency [rad/s]')
        fig.suptitle('Bode Magnitude Plot')
        plt.show()


        # Compare time responses for open- and closed-loop.
        t, u = disturbance(x=True)
        t, y = control.forced_response(model, T=t, U=u)
        t, y_clx = control.forced_response(clx, T=t, U=u[1:, :])

        plt.figure()
        plt.title("LQG control")
        plt.plot(t, y[0, :], label='open')
        plt.plot(t, y_clx[0, :], label='closed')
        plt.xlabel('t [s]')
        plt.ylabel('$\delta_x$ [m]')
        plt.legend()
        plt.show()
    


if __name__ == '__main__':
    # Build the models
    model_x = open_loop_x()
    model_y = open_loop_y()

    # Apply LQR control to the x and y model
    Kx, tf_x = LQR(model_x, x=True, y=False, plot=True)
    Kx, tf_x = LQR(model_y, x=False, y=True, plot=True)

    # Apply Kalman filter to the x model
    kalman(model_x, plot=True)

    # Apply LQG control to the x model
    LQG(model_x, plot=True)