import numpy as np
import matplotlib.pyplot as plt
import sys, os

# MPC library and dependencies
import casadi as cas
import do_mpc

from model import model
from controller import mpc

# All states can be directly measured
estimator = do_mpc.estimator.StateFeedback(model)

#######################
#    Simulator        #
#######################
simulator = do_mpc.simulator.Simulator(model)

params_simulator = {
    'integration_tool': 'cvodes',
    'abstol': 1e-2,
    'reltol': 1e-2,
    't_step': 50.0/3600.0,
    'integration_opts': {
        'linear_solver': 'csparse',
        'max_num_steps': 10
    }
}

simulator.set_param(**params_simulator)

## Uncertain parameters - create type of uncertainty for simulation
p_num = simulator.get_p_template()
tvp_num = simulator.get_tvp_template()
p_num['delH_R'] = 950 * np.random.uniform(0.75,1.25)
p_num['k_0'] = 7 * np.random.uniform(0.75*1.25)
def p_fun(t_now):
    return p_num
simulator.set_p_fun(p_fun)

simulator.setup()


#######################
#    Closed-loop      #
#######################
## Set the initial state of the controller and simulator:

# assume nominal values of uncertain parameters as initial guess
delH_R_real = 950.0
c_pR = 5.0
x0 = simulator.x0

x0['m_W'] = 10000.0
x0['m_A'] = 853.0
x0['m_P'] = 26.5

x0['T_R'] = 90.0 + 273.15
x0['T_S'] = 90.0 + 273.15
x0['Tout_M'] = 90.0 + 273.15
x0['T_EK'] = 35.0 + 273.15
x0['Tout_AWT'] = 35.0 + 273.15
x0['accum_monom'] = 300.0
x0['T_adiab'] = x0['m_A']*delH_R_real/((x0['m_W'] + x0['m_A'] + x0['m_P']) * c_pR) + x0['T_R']

mpc.x0 = x0
simulator.x0 = x0
estimator.x0 = x0

mpc.set_initial_guess()

print(simulator.model.model_type)

for k in range(100):
    print('1')
    u0 = mpc.make_step(x0)
    print('2')
    y_next = simulator.make_step(u0)
    print('3')
    x0 = estimator.make_step(y_next)

#######################
#      Animation      #
#######################
mpc_graphics = do_mpc.graphics.Graphics(mpc.data)

from matplotlib import rcParams
rcParams['axes.grid'] = True
rcParams['font.size'] = 18

fig, ax = plt.subplots(5, sharex=True, figsize=(16,12))
plt.ion()
# Configure plot:
mpc_graphics.add_line(var_type='_x', var_name='T_R', axis=ax[0])
mpc_graphics.add_line(var_type='_x', var_name='accum_monom', axis=ax[1])
mpc_graphics.add_line(var_type='_u', var_name='m_dot_f', axis=ax[2])
mpc_graphics.add_line(var_type='_u', var_name='T_in_M', axis=ax[3])
mpc_graphics.add_line(var_type='_u', var_name='T_in_EK', axis=ax[4])

ax[0].set_ylabel('T_R [K]')
ax[1].set_ylabel('acc. monom')
ax[2].set_ylabel('m_dot_f')
ax[3].set_ylabel('T_in_M [K]')
ax[4].set_ylabel('T_in_EK [K]')
ax[4].set_xlabel('time')

fig.align_ylabels()

from matplotlib.animation import FuncAnimation, ImageMagickWriter
def update(t_ind):
    print('Writing frame: {}.'.format(t_ind), end='\r')
    mpc_graphics.plot_results(t_ind=t_ind)
    mpc_graphics.plot_predictions(t_ind=t_ind)
    mpc_graphics.reset_axes()
    lines = mpc_graphics.result_lines.full
    return lines

n_steps = mpc.data['_time'].shape[0]


anim = FuncAnimation(fig, update, frames=n_steps, blit=True)

gif_writer = ImageMagickWriter(fps=5)
anim.save('anim_poly_batch.gif', writer=gif_writer)