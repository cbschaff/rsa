import pygame
from pygame.locals import *
import OpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import sys


class DroneRenderer(object):
    def __init__(self):
        self.q = gluNewQuadric()
        self.camera_xyz = [0., 0., 0.]
        self.camera_rpy = [0., 0., 0.]
        self.color1 = [0.7, 0.7, 0.7]
        self.color2 = [0.4, 0.4, 0.4]
        self.ground_color1 = [0.2, 1.0, 0.2]
        self.ground_color2 = [1.0, 0.2, 0.2]
        self.target = None
        self.initialized = False

    def _init_window(self):
        pygame.init()
        display = (800, 600)
        pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
        glEnable(GL_DEPTH_TEST);
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45.0, display[0]/display[1], 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        self.initialized = True

    def set_camera(self, xyz, rpy):
        self.camera_xyz = xyz
        self.camera_rpy = rpy

    def set_target(self, xyz, radius):
        self.target = (xyz, radius)

    def camera_transform(self):
        x, y, z = self.camera_xyz
        ar, ap, ay = self.camera_rpy
        glRotatef(-ar,1.0,0.0,0.0)
        glRotatef(-ap,0.0,1.0,0.0)
        glRotatef(-ay,0.,0.0,1.0)
        glTranslatef(-x, -y, -z)

    def draw_target(self, xyz, radius):
        glLoadIdentity()
        self.camera_transform()
        glTranslatef(*xyz)
        glColor3fv((1.0, 0., 0.))
        gluSphere(self.q, radius, 32, 32)

    def _apply_rpy(self, rpy):
        r, p, y = rpy
        axes = np.eye(3)
        axes[:2] = R.from_rotvec(y * axes[2]).apply(axes[:2])
        axes[:1] = R.from_rotvec(p * axes[1]).apply(axes[:1])
        glRotatef(180. / np.pi * r, axes[0,0], axes[0,1], axes[0,2])
        glRotatef(180. / np.pi * p, axes[1,0], axes[1,1], axes[1,2])
        glRotatef(180. / np.pi * y, axes[2,0], axes[2,1], axes[2,2])

    def draw_body(self, pos, orientation, radius):
        glLoadIdentity()
        self.camera_transform()
        glColor3fv(self.color1)
        glTranslatef(pos[0], pos[1], pos[2])
        # self._apply_rpy(orientation)
        # glTranslatef(0., 0., -radius)
        # gluCylinder(self.q, radius, radius, 2*radius, 32, 32)
        gluSphere(self.q, radius, 32, 32)

    def draw_arm(self, arm, pos, orientation, length, rod_radius,
                  rotor_radius):
        glLoadIdentity()
        if arm == 0:
            glColor3fv(self.color2)
        else:
            glColor3fv(self.color1)
        self.camera_transform()
        glTranslatef(pos[0], pos[1], pos[2])
        self._apply_rpy(orientation)
        glRotatef(90*arm,0.0,0.0,1.0)
        glRotatef(90,0.0,1.0,0.0)
        gluCylinder(self.q, rod_radius, rod_radius, length, 32, 32)

        glLoadIdentity()
        self.camera_transform()
        glTranslatef(pos[0], pos[1], pos[2])
        self._apply_rpy(orientation)
        glRotatef(90*arm,0.0,0.0,1.0)
        glTranslatef(length, 0., -3*rod_radius)
        gluCylinder(self.q, rotor_radius, 0., 4*rod_radius, 32, 32)

    def draw_ground(self, boundx=100, boundy=100):
        glLoadIdentity()
        self.camera_transform()
        glBegin(GL_LINES)
        for x in range(-100, 100):
            if np.abs(x) >= boundx:
                glColor3fv(self.ground_color2)
                glVertex3fv((x, -100., 0.))
                glVertex3fv((x, 100., 0.))
            else:
                glColor3fv(self.ground_color2)
                glVertex3fv((x, -100., 0.))
                glVertex3fv((x, -boundx, 0.))
                glVertex3fv((x, boundx, 0.))
                glVertex3fv((x, 100., 0.))
                glColor3fv(self.ground_color1)
                glVertex3fv((x, -boundx, 0.))
                glVertex3fv((x, boundx, 0.))
        for y in range(-100, 100):
            if np.abs(y) >= boundy:
                glColor3fv(self.ground_color2)
                glVertex3fv((-100., y, 0.))
                glVertex3fv((100., y, 0.))
            else:
                glColor3fv(self.ground_color2)
                glVertex3fv((-100., y, 0.))
                glVertex3fv((-boundx, y, 0.))
                glVertex3fv((boundx, y, 0.))
                glVertex3fv((100., y, 0.))
                glColor3fv(self.ground_color1)
                glVertex3fv((-boundx, y, 0.))
                glVertex3fv((boundx, y, 0.))
        glEnd()

    def draw_world(self, pos, orientation, radius, length, bounds=(100, 100)):
        if not self.initialized:
            self._init_window()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        rotor_radius = (length - radius) / 2
        rod_radius = radius / 10.

        self.draw_ground(*bounds)
        if self.target:
            self.draw_target(*self.target)
        for i in range(4):
            self.draw_arm(i, pos, orientation, length, rod_radius, rotor_radius)
        self.draw_body(pos, orientation, radius)
        pygame.display.flip()

    def get_pixels(self):
        data = glReadPixels(0, 0, 800, 600, GL_RGB, GL_UNSIGNED_BYTE)
        return np.frombuffer(data, dtype=np.uint8).reshape(600, 800, 3)[::-1]


class DroneFollower(object):
    def __init__(self, offset, alpha_pos, alpha_rot):
        self.offset = np.array(offset)
        self.alpha_pos = alpha_pos
        self.alpha_rot = alpha_rot

        self.camera_pos = np.zeros(3)
        self.camera_angle = np.zeros(3)

    def _get_angle_diff(self, a1, a2):
        d = a2 - a1
        for i, dd in enumerate(d):
            if dd < -180.:
                d[i] += 360.
            elif dd >= 180.:
                d[i] -= 360.
        return d

    def _get_rotation_from_vecs(self, v1, v2):
        v1 = v1 / np.sqrt(np.sum(v1 ** 2))
        v2 = v2 / np.sqrt(np.sum(v2 ** 2))
        rot_axis = np.cross(v1, v2)
        if np.allclose(rot_axis, 0.):
            return R.from_euler('xyz', [0., 0., 0.])
        rot_axis /= np.sqrt((rot_axis ** 2).sum())
        angle = np.arccos(v1.dot(v2))
        return R.from_rotvec(angle * rot_axis)

    def _look_at_drone(self, camera_pos, drone_pos, drone_ori):
        dir = drone_pos - camera_pos
        dir /= np.sqrt(np.sum(dir ** 2))
        camera_start = np.array([0., 0., -1.])
        rot = self._get_rotation_from_vecs(camera_start, dir)

        # orient camera up
        camera_up = rot.apply(np.array([0., 1., 0.]))
        z_body = self._apply_rpy(np.array([0., 0., -1.]), drone_ori)

        # project z_body onto dir
        proj = z_body.dot(dir) * dir

        # find vector orthogonal to dir in the plane of (z_body, dir)
        dir_ortho = z_body - proj
        dir_ortho /= np.sqrt(np.sum(dir_ortho ** 2))

        # find angle between dir_ortho and rotated camera_up
        # and the direction to rotate it
        angle = np.arccos(camera_up.dot(dir_ortho))
        sign = np.sign(np.cross(dir_ortho, camera_up).dot(dir))

        # combine rotations for looking at drone and orienting camera
        rot_up = rot.from_rotvec(- sign * angle * dir)
        rot_tot = rot_up * rot
        return rot_tot.as_euler('xyz', degrees=True)

    def _apply_rpy(self, v, rpy):
        xyz = np.eye(3)
        yaw = R.from_rotvec(rpy[2] * xyz[2])
        xyz[:2] = yaw.apply(xyz[:2])
        pitch = R.from_rotvec(rpy[1] * xyz[1])
        xyz[:1] = pitch.apply(xyz[:1])
        roll = R.from_rotvec(rpy[0] * xyz[0])
        v = yaw.apply(v)
        v = pitch.apply(v)
        v = roll.apply(v)
        return v

    def _set_camera_pose(self, drone_pos, drone_ori, alpha_pos, alpha_ori):
        drone_pos = np.array(drone_pos)
        drone_ori = np.array(drone_ori)
        cp = drone_pos + self._apply_rpy(self.offset, drone_ori)
        self.camera_pos = alpha_pos * self.camera_pos + (1.0 - alpha_pos) * cp
        ca = self._look_at_drone(self.camera_pos, drone_pos, drone_ori)
        angle_diff = self._get_angle_diff(self.camera_angle, ca)
        self.camera_angle += (1.0 - alpha_ori) * angle_diff
        for i, c in enumerate(self.camera_angle):
            if c < -180.:
                self.camera_angle[i] += 360.
            elif c >= 180.:
                self.camera_angle[i] -= 360.

    def reset(self, drone_pos, drone_ori):
        self._set_camera_pose(drone_pos, drone_ori, 0.0, 0.0)
        return self.camera_pos, self.camera_angle

    def step(self, drone_pos, drone_ori):
        self._set_camera_pose(drone_pos, drone_ori, self.alpha_pos,
                              self.alpha_rot)
        return self.camera_pos, self.camera_angle


def main():
    from residual_shared_autonomy.drone_sim import Drone
    np.random.seed(0)

    d = Drone(0., 0., 0., 0., 0., 0., 0.,
              0., 0., 0., 0., 0., 6, 0.5, 1, 2)
    torques = np.zeros(3)
    forces = np.array([0., 0., -200.])
    states = []
    r = DroneRenderer()
    f = DroneFollower([-10., 0., -10.], 0.9, 0.9)
    r.set_target([50., 0., 0.], 1.)
    cp, ca = f.reset(d.pos_i, d.ori_v)

    for _ in range(100):
        r.set_camera(cp, ca)
        r.draw_world(d.pos_i, d.ori_v, d.r_center, d.lengh)
        torques = 50 * np.random.uniform(-1., 1., (3,))
        s = d.step(torques, forces, 1./50.)
        states.append(s)
        cp, ca = f.step(d.pos_i, d.ori_v)
        time.sleep(0.02)
    h_off = states[-1][2]
    while states[-1][2] - 1e-5 < h_off:
        r.set_camera(cp, ca)
        r.draw_world(d.pos_i, d.ori_v, d.r_center, d.lengh)
        s = d.step(0*torques, 0*forces, 1./50.)
        states.append(s)
        cp, ca = f.step(d.pos_i, d.ori_v)
        time.sleep(0.02)
    for _ in range(100):
        r.set_camera(cp, ca)
        r.draw_world(d.pos_i, d.ori_v, d.r_center, d.lengh)
        torques = 50 * np.random.uniform(-1., 1., (3,))
        s = d.step(torques, forces, 1./50.)
        states.append(s)
        cp, ca = f.step(d.pos_i, d.ori_v)
        time.sleep(0.02)

    import matplotlib
    matplotlib.use('TkAgg')
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


if __name__ == '__main__':
    main()
