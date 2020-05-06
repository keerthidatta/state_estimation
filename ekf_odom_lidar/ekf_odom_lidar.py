import pickle
import numpy as np
from numpy.linalg import inv

import matplotlib.pyplot as plt

with open('data/data.pickle', 'rb') as f:
    data = pickle.load(f)
    
print(data)

t = data['t']  # timestamps [s]

x_init  = data['x_init'] # initial x position [m]
y_init  = data['y_init'] # initial y position [m]
th_init = data['th_init'] # initial theta position [rad]

# input signal
v  = data['v']  # translational velocity input [m/s]
om = data['om']  # rotational velocity input [rad/s]

# bearing and range measurements, LIDAR constants
b = data['b']  # bearing to each landmarks center in the frame attached to the laser [rad]
r = data['r']  # range measurements [m]
l = data['l']  # x,y positions of landmarks [m]
d = data['d']  # distance between robot center and laser rangefinder [m]

v_var = 0.01  # translation velocity variance  
om_var = 1  # rotational velocity variance 
r_var = 0.01  # range measurements variance
b_var = 0.01  # bearing measurement variance

Q_km = np.diag([v_var, om_var]) # input noise covariance 
cov_y = np.diag([r_var, b_var])  # measurement noise covariance 

x_est = np.zeros([len(v), 3])  # estimated states, x, y, and theta
P_est = np.zeros([len(v), 3, 3])  # state covariance matrices

x_est[0] = np.array([x_init, y_init, th_init]) # initial state
P_est[0] = np.diag([1, 1, 0.1]) # initial state covariance

# Wraps angle to (-pi,pi] range
def wraptopi(x):
    if x > np.pi:
        x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
    elif x < -np.pi:
        x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
    return x

def measurement_update(lk, rk, bk, P_check, x_check):
    
    x = x_check[0, 0]
    y = x_check[1, 0]
    theta = wraptopi(x_check[2, 0])
    lx = lk[0]
    ly = lk[1]
    bk = wraptopi(bk)
    
    #Debug - parameters:
    print("##########Function Parameters##########")
    print('x:',x)
    print('y:',y)
    print('theta:',theta)
    print('lx:',lx)
    print('ly:',ly)
    print('rk:',rk)
    print('bk:', bk)
    print('covariance P:', P_check)
    
    # 1. Compute measurement Jacobian
    
    #Derivative with respect to x
    d_x0 = (x - lx)/(np.sqrt((lx - x)**2 + (ly - y)**2))
    d_x1 = (ly - y)/((lx - x)**2 + (ly - y)**2)
    
    print('d_x0:', d_x0)
    print('d_x1:', d_x1)
    
    #Derivative with respect to y
    d_y0 = (y - ly)/(np.sqrt((lx - x)**2 + (ly - y)**2))
    d_y1 = (x - lx)/((lx - x)**2 + (ly - y)**2)
    
    print('d_y0:', d_y0)
    print('d_y1:', d_y1)
    
    #Derivative with respect to theta:
    d_theta0 = 0
    d_theta1 = -1
    
    print('d_theta0:', d_theta0)
    print('d_theta1:', d_theta1)
    
    #Compute H matrix:
    
    #H = np.stack((np.concatenate((d_x0, d_y0, d_theta0),axis=0), np.concatenate((d_x1, d_y1, d_theta1),axis=0)))
    H = np.array([[d_x0, d_y0, d_theta0], [d_x1, d_y1, d_theta1]])
    print('H:', H)
    
    # 2. Compute Kalman Gain
    M = np.eye(2, dtype=int)
    R = cov_y
    
    K = P_check @ H.T @ inv(H @ P_check @ H.T + M @ R @ M.T)
    
    print('K:', K)

    # 3. Correct predicted state (remember to wrap the angles to [-pi,pi])
    Measured = np.array([[rk, bk]]).T
    print('Measured:', Measured)
    print('Measured shape:', np.shape(Measured))
    
    print('Numerator:theta:',np.arctan2(ly-y, lx-x))
    print('denom:theta:', theta)
    print('theta: together:', np.arctan2(ly-y, lx-x) - theta)
    print('wrappped:', wraptopi(np.arctan2(ly-y, lx-x) - theta))
    
    #expected = np.concatenate(((np.sqrt((lx - x)**2 + (ly - y)**2)), wraptopi(wraptopi(np.arctan2(ly-y, lx-x)) - theta)),axis=0)
    expected = np.array([[np.sqrt((lx - x)**2 + (ly - y)**2)], [wraptopi(wraptopi(np.arctan2(ly-y, lx-x)) - theta)]])

        
    print('Expecte:', expected)
    print('Expected shape:', np.shape(expected))
    
    #x_check = np.concatenate(([x, y, theta]), axis=0) + K @ (Measured - expected)
    x_check = np.array([[x, y, theta]]).T + K @ (Measured - expected)
    print('x_check:', x_check)
    print('x_check shape:', np.shape(x_check))

    # 4. Correct covariance

    P_check = (np.eye(3, dtype=int) - K@H) @ P_check
    
    print('P_check:', P_check)
    print('shape p:', np.shape(P_check))
    return x_check, P_check

#### 5. Main Filter Loop #######################################################################
for k in range(1, len(t)):  # start at 1 because we've set the initial prediciton
   
    delta_t = t[k] - t[k - 1]  # time step (difference between timestamps)

    x = x_est[k-1, 0]
    y = x_est[k-1, 1]
    theta = wraptopi(x_est[k-1, 2])
    
    # 1. Update state with odometry readings (remember to wrap the angles to [-pi,pi])
    B = np.array([[np.cos(theta), 0], [np.sin(theta), 0], [0, 1]])
    x_check = np.array([[x, y, theta]]).T + B @ np.array([[v[k], wraptopi(om[k])]]).T
    
    print('x_check:', x_check)
    print('x_check shape:', np.shape(x_check))

    # 2. Motion model jacobian with respect to last state
    F = np.array([
        [1, 0, wraptopi(-np.sin(theta)* (v[k]))],
        [0, 1, wraptopi(-np.cos(theta)*(v[k]))],
        [0, 0, wraptopi(1+(1)*1)]
    ])
    
    print('F:', F)
    
    #F_km = np.zeros([3, 3])

    # 3. Motion model jacobian with respect to noise
    L = np.array([
        [np.cos(theta), np.sin(theta), 0],
    [0, 0, 1]
    ]).T
    #L_km = np.zeros([3, 2])

    # 4. Propagate uncertainty
    P_check = F @ P_est[k-1] @ F.T + L @ Q_km @ L.T
    
    #x_check, P_check = measurement_update(l[5], r[1, 0], b[1, 0],np.diag([1, 1, 0.1]) , np.array([[10, 5, 1.5]]).T)
    
    # 5. Update state estimate using available landmark measurements
    for i in range(len(r[k])):
        x_check, P_check = measurement_update(l[i], r[k, i], b[k, i], P_check, x_check)

    # Set final state predictions for timestep
    x_est[k, 0] = x_check[0]
    x_est[k, 1] = x_check[1]
    x_est[k, 2] = x_check[2]
    P_est[k, :, :] = P_check

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(x_est[:, 0], x_est[:, 1])
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_title('Estimated trajectory')
plt.show()

e_fig = plt.figure()
ax = e_fig.add_subplot(111)
ax.plot(t[:], x_est[:, 2])
ax.set_xlabel('Time [s]')
ax.set_ylabel('theta [rad]')
ax.set_title('Estimated trajectory')
plt.show()