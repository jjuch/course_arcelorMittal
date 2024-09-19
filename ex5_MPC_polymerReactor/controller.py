import numpy as np
import matplotlib.pyplot as plt
import sys, os

# MPC library and dependencies
import casadi as cas
import do_mpc

from model import model

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 5,
    'n_robust': 1,
    'open_loop': 0,
    't_step': 50.0/3600.0,
    'state_discretization': 'collocation',
    'collocation_type': 'radau',
    'collocation_deg': 2,
    'collocation_ni': 2,
    'store_full_solution': True,
    # Use MA27 linear solver in ipopt for faster calculations:
    #'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
}

mpc.set_param(**setup_mpc)

## OBJECTIVE
# Produce as fast as possible 20680kg of m_P
# Has to be smooth -> penalty on input changes
_x = model.x
mterm = - _x['m_P'] # Terminal cost
lterm = - _x['m_P'] # Stage cost

mpc.set_objective(mterm=mterm, lterm=lterm) # Optimal Control Problem: sum(m_P[k], k=0..N+1)
mpc.set_rterm(m_dot_f=0.002, T_in_M=0.004, T_in_EK=0.002) # Penalty on control input changes

## CONSTRAINTS
# On states
# Desired reaction temperature 90°C +- 2°C

# auxiliary term
temp_range = 2.0

# lower bound states
mpc.bounds['lower','_x','m_W'] = 0.0
mpc.bounds['lower','_x','m_A'] = 0.0
mpc.bounds['lower','_x','m_P'] = 26.0

mpc.bounds['lower','_x','T_R'] = 363.15 - temp_range
mpc.bounds['lower','_x','T_S'] = 298.0
mpc.bounds['lower','_x','Tout_M'] = 298.0
mpc.bounds['lower','_x','T_EK'] = 288.0
mpc.bounds['lower','_x','Tout_AWT'] = 288.0
mpc.bounds['lower','_x','accum_monom'] = 0.0

# upper bound states
mpc.bounds['upper','_x','T_S'] = 400.0
mpc.bounds['upper','_x','Tout_M'] = 400.0
mpc.bounds['upper','_x','T_EK'] = 400.0
mpc.bounds['upper','_x','Tout_AWT'] = 400.0
mpc.bounds['upper','_x','accum_monom'] = 30000.0
mpc.bounds['upper','_x','T_adiab'] = 382.15

# The upper bound of the reactor temperature is set via a soft-constraint
mpc.set_nl_cons('T_R_UB', _x['T_R'], ub=363.15+temp_range, soft_constraint=True, penalty_term_cons=1e4)

# On inputs
# lower bound inputs
mpc.bounds['lower','_u','m_dot_f'] = 0.0
mpc.bounds['lower','_u','T_in_M'] = 333.15
mpc.bounds['lower','_u','T_in_EK'] = 333.15

# upper bound inputs
mpc.bounds['upper','_u','m_dot_f'] = 3.0e4
mpc.bounds['upper','_u','T_in_M'] = 373.15
mpc.bounds['upper','_u','T_in_EK'] = 373.15

## SCALING
# Scale to optimise without bias
# states
mpc.scaling['_x','m_W'] = 10
mpc.scaling['_x','m_A'] = 10
mpc.scaling['_x','m_P'] = 10
mpc.scaling['_x','accum_monom'] = 10

# control inputs
mpc.scaling['_u','m_dot_f'] = 100

## UNCERTAIN VALUES
#Delta H_R and k_0 can vary +-30%
delH_R_var = np.array([950.0, 950.0 * 1.30, 950.0 * 0.70])
k_0_var = np.array([7.0 * 1.00, 7.0 * 1.30, 7.0 * 0.70])

mpc.set_uncertainty_values(delH_R = delH_R_var, k_0 = k_0_var)

mpc.setup()