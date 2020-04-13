import numpy as np


class Drone(object):
    """Drone state."""

    def __init__(self, p_n, p_e, h, u, v, w, phi, theta, psi, p, q, r,
                 m_center, r_center, m_rotor, l):
        """State and dynamics of the drone.

        Notation and equations from "Quadrotor Dynamics and Control Rev 0.1"
        by Randal W. Beard.
        https://scholarsarchive.byu.edu/cgi/viewcontent.cgi?article=2324&context=facpub

        The drone is assumed to consist of a spherical dense center of mass
        m_center and radius r_center and 4 point mass rotors with mass m_rotor
        located a distance of l from the center.

        position in inertial frame
        --------------------------
        p_n: position north measured along i_i
        p_e: position east measured along j_i
        h: height measured along k_i

        velocity in body frame
        ----------------------
        u: velocity measured along i_b
        v: velocity measured along j_b
        w: velocity measured along k_b

        orientation in vehicle, vehicle_1, and vehicle_2 frames
        -------------------------------------------------------
        phi: roll angle in vehicle_2 frame
        theta: pitch angle in vehicle_1 frame
        psi: yaw angle in vehicle frame

        angular rate in body frame
        --------------------------
        p: roll rate measured along i_b
        q: pitch rate measured along j_b
        r: yaw rate measured along k_b

        drone parameters
        ----------------
        m_center: mass of the center of the drone
        r_center: radius of the center of the drone
        m_rotor: mass of each rotor
        l: distance of rotors from center
        """
        self.pos_i = np.array([p_n, p_e, h])
        self.vel_b = np.array([u, v, w])
        self.ori_v = np.array([phi, theta, psi])
        self.angvel_b = np.array([p, q, r])

        self.m = m_center + 4 * m_rotor
        self.m_center = m_center
        self.m_rotor = m_rotor
        self.r_center = r_center
        self.lengh = l

        # compute moment of inertia
        self.jx = 2 * m_center * (r_center ** 2) / 5 + 2 * (l ** 2) * m_rotor
        self.jy = 2 * m_center * (r_center ** 2) / 5 + 2 * (l ** 2) * m_rotor
        self.jz = 2 * m_center * (r_center ** 2) / 5 + 4 * (l ** 2) * m_rotor

        # compute useful constants
        self._jc = np.array([(self.jy - self.jz) / self.jx,
                             (self.jz - self.jx) / self.jy,
                             (self.jx - self.jy) / self.jz])
        self._jinv = np.array([1/self.jx, 1/self.jy, 1/self.jz])
        self._g = 9.80665

    def state(self):
        """Get the state of the drone.

        State consists of position, velocity, orientation, and angular velocity.
        """
        return np.concatenate([self.pos_i, self.vel_b, self.ori_v,
                               self.angvel_b])

    def step(self, torques, f, time_step):
        """Apply forces and torques for time_step amount of time.

        torques: a numpy array of roll, pitch, yaw torques in the body frame.
        f: a numpy array of the x, y, z forces in the body frame,
           excluding gravity.
        time_step: the step size for approximating new state.
        """
        c = np.cos(self.ori_v)
        s = np.sin(self.ori_v)
        t = np.tan(self.ori_v)
        p, q, r = self.angvel_b
        u, v, w = self.ori_v
        g = self._g

        # compute velocity in inertial frame
        rot_b_i = np.array([
            [c[1]*c[2], s[0]*s[1]*c[2] - c[0]*s[2], c[0]*s[1]*c[2] + s[0]*s[2]],
            [c[1]*s[2], s[0]*s[1]*s[2] + c[0]*c[2], c[0]*s[1]*s[2] - s[0]*c[2]],
            [-s[1],     s[0]*c[1],                  c[0]*c[1]]
        ])
        vel_i = rot_b_i.dot(self.vel_b)

        # compute acceleration in body frame
        acc_b = (np.array([r*v - q*w, p*w - r*u, q*u - p*v])
                 + np.array([-g*s[1], g*c[1]*s[0], g*c[1]*c[0]])
                 + f / self.m)

        # compute angular velocity in vehicle frames
        rot_b_v = np.array([
            [1., s[0]*t[1], c[0]*t[1]],
            [0., c[0], -s[0]],
            [0., s[0]/c[1], c[0]/c[1]]
        ])
        angvel_v = rot_b_v.dot(self.angvel_b)

        # compute angular acceleration in body frame
        angacc_b = self._jc * np.array([q*r, p*r, p*q]) + self._jinv * torques

        # update state
        self.pos_i += time_step * vel_i
        self.vel_b += time_step * acc_b
        self.ori_v += time_step * angvel_v
        self.angvel_b += time_step * angacc_b
        self.ori_v = ((self.ori_v + np.pi) % (2 * np.pi)) - np.pi
        return self.state()

    def rotate_i_to_b(self, v):
        c = np.cos(self.ori_v)
        s = np.sin(self.ori_v)
        rot_b_i = np.array([
            [c[1]*c[2], s[0]*s[1]*c[2] - c[0]*s[2], c[0]*s[1]*c[2] + s[0]*s[2]],
            [c[1]*s[2], s[0]*s[1]*s[2] + c[0]*c[2], c[0]*s[1]*s[2] - s[0]*c[2]],
            [-s[1],     s[0]*c[1],                  c[0]*c[1]]
        ])
        return rot_b_i.T.dot(v)



if __name__ == '__main__':
    # Test the simulator by applying random torques for roll and pitch
    # with a constant upward force, then turning off all motors for a bit,
    # then continuing.
    d = Drone(0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 1., 1., 2.)
    print(d.state())
    torques = np.zeros(3)
    forces = np.array([0., 0., -200.])
    states = []
    for _ in range(100):
        torques = np.random.uniform(-1., 1., (3,))
        s = d.step(torques, forces, 1./50.)
        print(s)
        states.append(s)
    h_off = states[-1][2]
    while states[-1][2] - 1e-5 < h_off:
        s = d.step(0*torques, 0*forces, 1./50.)
        print(s)
        states.append(s)
    for _ in range(100):
        torques = np.random.uniform(-1., 1., (3,))
        s = d.step(torques, forces, 1./50.)
        print(s)
        states.append(s)

    from matplotlib import pyplot as plt
    states = np.array(states)
    x = np.arange(states.shape[0]) / 50.
    names = ['x', 'y', 'h', 'u', 'v', 'w']
    names2 = ['phi', 'theta', 'psi', 'p', 'q', 'r']
    f, (ax1, ax2) = plt.subplots(1, 2)
    for i, n in enumerate(names):
        ax1.plot(x, states[:, i], label=n)
    ax1.legend()
    for i, n in enumerate(names2):
        ax2.plot(x, states[:, 6 + i], label=n)
    ax2.legend()
    plt.show()
