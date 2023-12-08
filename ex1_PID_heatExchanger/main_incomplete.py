from utils import leesCSV, plotData, raaklijn
from heatExchanger_model import Gp

import numpy as np
import control

#### 1. Benader stapantwoord
# Lees csv
t, data = leesCSV('heatExchanger_measurement.csv', plot=True)

# Eerste orde plus dode tijd
stap_grootte = 10
t_start = 0 # Wanneer is de stap aangelegd
K = 
tau = 
L = 

print("""
   {:.2f}
---------- exp(-{:.2f}s)
{:.2f}s + 1
""".format(K, tau, L))

# Maak overdrachtsfunctie
FOPDT = control.tf([K], [tau, 1])
T, yout = control.step_response(FOPDT)
yout = yout * stap_grootte
T = T + t_start + L
plotData([t, T], [data, yout])

#### 2. A. Stel PI af (zie Ziegler-Nichols)
Kc = 
Ti = 
PI = control.tf([Kc*Ti, Kc], [Ti, 0])

PI_GK = control.feedback(PI * Gp)
t_PI, y_PI = control.step_response(PI_GK)
y_PI = y_PI * stap_grootte

#### 2. B. Stel PID af (zie Ziegler-Nichols)
Kc = 
Ti = 
Td = 
PID = control.tf([Kc*Ti*Td, Kc*Ti, Kc], [Ti, 0])

PID_GK = control.feedback(PID * Gp)
t_PID, y_PID = control.step_response(PID_GK)
y_PID = y_PID * stap_grootte

# Plotten
plotData([t, t_PI, t_PID], [data, y_PI, y_PID], labels=['meting', 'PI', 'PID'])