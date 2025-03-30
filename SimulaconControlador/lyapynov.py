import numpy as np
import matplotlib.pyplot as plt

# Vector de tiempo, ts 10 ms 
ts = 0.01
t = np.arange(0, 35 + ts, ts)
Q = len(t)

scaleRobot = 1

# Trayectoria, posiciones x,y del mapa ocupacional 
seq_desplazada = np.array([
    [0, 2], [1, 1], [2, 1], [3, 1], [4, 2], [5, 3], [6, 4],
    [7, 5], [8, 6], [8, 7], [8, 8], [9, 9]
])
# Ajustamos el offset para poder visualizar. 
desplazamiento = -seq_desplazada[0]
seq_desplazada = seq_desplazada + desplazamiento
destinations = seq_desplazada

# Variables para almacenar datos
# Energia 
k_u = 1
k_w = 1
power = np.zeros(Q)

# Posicion
x1 = np.zeros(Q + 1)
y1 = np.zeros(Q + 1)

# Offset de efector
a = 0.1
hx = np.zeros(Q + 1)
hy = np.zeros(Q + 1)
phi = np.zeros(Q + 1)

uRef = np.zeros(Q)
wRef = np.zeros(Q)

hxe = np.zeros(Q)
hye = np.zeros(Q)

# Controlador 
for k in range(Q):
    current_dest = destinations[min(k * len(destinations) // Q, len(destinations) - 1)]
    
    # Estimacion del error 
    hxe[k] = current_dest[0] - hx[k]
    hye[k] = current_dest[1] - hy[k]
    he = np.array([hxe[k], hye[k]])
    
    # Matriz Jacobiana  
    J = np.array([
        [np.cos(phi[k]), -a * np.sin(phi[k])],
        [np.sin(phi[k]), a * np.cos(phi[k])]
    ])
    
    # Parámetros de control
    K = np.array([[0.5, 0], [0, 0.5]])
    
    # Ley de control. 
    qpref = np.linalg.pinv(J) @ (K @ np.tanh(he))
    
    # Acciones de control para cada rueda
    uRef[k] = qpref[0]
    wRef[k] = qpref[1]
    
    # Estimación de potencia
    power[k] = k_u * uRef[k]**2 + k_w * wRef[k]**2
    
    # Actualizacion de parametros del robot
    phi[k + 1] = phi[k] + ts * wRef[k]
    x1p = uRef[k] * np.cos(phi[k + 1])
    y1p = uRef[k] * np.sin(phi[k + 1])
    
    x1[k + 1] = x1[k] + ts * x1p
    y1[k + 1] = y1[k] + ts * y1p
    
    hx[k + 1] = x1[k + 1] + a * np.cos(phi[k + 1])
    hy[k + 1] = y1[k + 1] + a * np.sin(phi[k + 1])

# Visualizacion 
plt.figure()
plt.plot(destinations[:, 0], destinations[:, 1], 'g--', linewidth=2, label='Desired trajectory')
plt.plot(hx, hy, 'b', linewidth=2, label='Estimated trajectory')
plt.plot(destinations[:, 0], destinations[:, 1], 'ro', markersize=8, markerfacecolor='r', label='Target points')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.legend()
plt.grid()
plt.axis('equal')
plt.show()

plt.figure()
plt.plot(t, power, linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Power Consumption (W)')
plt.title('Power Consumption over Time')
plt.grid()
plt.show()
