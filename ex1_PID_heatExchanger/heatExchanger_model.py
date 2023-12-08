"""
Warmtewisselaar model voor de Python opdracht van oefeningenles 5.

Tsi: de ingaande temperatuur van de stoom is de ingang.
Twu: de uitgaande temperatuur van het koude water is de uitgang.
Twi: de ingaande temperatuur van het koude water is een storing.

Gp: overdrachtsfuncties van ingang naar uitgang
Gd: overdrachtsfunctie van verstoring naar uitgang
"""

import control
import numpy as np
import matplotlib.pyplot as plt

# Numerieke waarden:
aw = 2.92
a_s = 5
vw = 1
vs = 2
Dz = 1/3
Aw = vw/Dz + aw
As = vs/Dz + a_s

## Gp(s)
# Noemer Np
c6 = 1
c5 = 3*Aw + 3*As
c4 = 3*Aw**2 + 3*As**2 + 9*As*Aw - 3*a_s*aw
c3 = Aw**3 + As**3 + 9*As*Aw**2 + 9*As**2*Aw - 6*a_s*aw*Aw - 6*a_s*aw*As
c2deel1 = 3*As*Aw**3 + 3*Aw*As**3 + 9*As**2*Aw**2
c2deel2 = -3*a_s*aw*(Aw**2 + As**2 + 4*Aw*As)
c2deel3 = 3*aw**2*a_s**2
c2 = c2deel1 + c2deel2 + c2deel3
c1deel1 = 3*As**2*Aw**3 + 3*Aw**2*As**3
c1deel2 = -3*a_s*aw*(2*As*Aw**2 + 2*Aw*As**2)
c1deel3 = 3*aw**2*a_s**2*(As + Aw)
c1 = c1deel1 + c1deel2 + c1deel3
c0 = As**3*Aw**3 - aw**3*a_s**3 - 3*a_s*aw*As**2*Aw**2 + 3*aw**2*a_s**2*Aw*As

Np = [c6, c5, c4, c3, c2, c1, c0]
polen = np.roots(Np)


# Teller Tp
t2 = vs/Dz*vw/Dz*aw + aw*(vs/Dz)**2 + aw*(vw/Dz)**2
t1 = vs/Dz*vw/Dz*aw*(Aw + As) + aw*(vs/Dz)**2*2*Aw + aw*(vw/Dz)**2*2*As
t0deel1 = vs/Dz*vw/Dz*aw*Aw*As + aw*(vs/Dz)**2*Aw**2
t0deel2 = aw*(vw/Dz)**2*As**2 + a_s*aw**2*(vw/Dz)*(vs/Dz)
t0 = t0deel1 + t0deel2
Tp = [t2, t1, t0]

# tf
Gp = control.tf(Tp, Np)

## Gd(s)
# Teller Td
d3 = (vw/Dz)**2
d2 = (vw/Dz)**2*3*As
d1 = (vw/Dz)**2*3*As**2 + 2*aw*a_s*(vs/Dz)*(vw/Dz) + aw*a_s*(vs/Dz)**2
d0 = (vw/Dz)**2*As**3 + 2*aw*a_s*(vs/Dz)*(vw/Dz)*As + aw*a_s*(vs/Dz)**2*Aw
Td = [d3, d2, d1, d0]

Gd = control.tf(Td, Np)


if __name__ == '__main__':
    print('Gp = ', Gp)
    print('Gd = ', Gd)

    # Stapantwoord Gp en Gd
    tijd = np.linspace(0, 5, 1000)
    tp, yp = control.step_response(Gp, tijd)
    # plotData(tp, yp, xlabel="tijd [min]", ylabel="uitgang [°C]", titel="Gp(s)")
    td, yd = control.step_response(Gd, tijd)

    plt.figure()
    plt.plot(tp, yp, label="Gp")
    plt.plot(td, yd, label="Gd")
    plt.xlabel('tijd [min]')
    plt.ylabel('uitgang [°C]')
    plt.legend()
    plt.show()
