import numpy as np 
from numpy.linalg import inv

#------------------------------------------------------
#Problem
#Using non linear angle measurement of a landmark from camera, compute position of vehicle in which camera is placed,

#u = acceleration
#d = distance from vehicle to landmark
#s = Heigt of the landmark
#p = position of the vehicle           
#Phi = tan-1(s/(d-p))
#measurement phi = 30 (y1)
#Data:

x_0 = np.array([[0, 5]]).T
p_0 = np.array([[0.01, 0], [0, 1]]).T

u_0 = -2
s = 20
d = 40
y1 = 30
delta_t = 0.5

#Compute : Prediction and correction for given measurement using EKF


F_k_1 = np.array([[1, delta_t],[0, 1]])
x_k_1 = x_0
G_k_1 = np.array([[0, delta_t]]).T

x_k = F_k_1 @ x_k_1 + G_k_1 * u_0 

L_k_1 = np.array([[1, 0], [0, 1]])
Q_k_1 = np.array([[0.1, 0],[0, 0.1]])
p_k_1 = p_0
p_k = F_k_1 @ p_k_1 @ F_k_1.T + L_k_1 @ Q_k_1 @ L_k_1

#correction:
p = x_k[0, 0]
d_phi = (s/((d-p)**2 + s**2))
H_k = np.array([[d_phi, 0]])
M_k = 1
R = 0.01
k_k = p_k @ H_k.T * inv(H_k @ p_k @ H_k.T + M_k * R * M_k)

#Phi = np.arctan(s/(d-p))

X_k = x_k + k_k * (y1 - 28.1) 

print(X_k)
P_k = (np.eye(2, dtype=int) - k_k @ H_k) @ p_k
print(P_k)