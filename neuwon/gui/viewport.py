import math
import numpy as np
import pygame
from pygame.locals import *
from scipy.spatial.transform import Rotation

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from neuwon.database import Database, Pointer
from neuwon.gui.primatives import *

epsilon = np.finfo(float).eps

class Scene:
    def __init__(self, database, lod=2.5):
        lod     = float(lod)
        if hasattr(database, "get_database"): database = database.get_database()
        Segment = database.get("Segment")
        self.num_seg = num_seg = len(Segment)
        objects = np.zeros(num_seg, dtype=object)
        for seg in Segment.get_all_instances():
            idx = seg.get_unstable_index()
            nslices = max(3, int(lod * seg.diameter))
            if seg.is_sphere():
                objects[idx] = Sphere(seg.coordinates, 0.5 * seg.diameter, nslices)
            elif seg.is_cylinder():
                objects[idx] = Cylinder(seg.coordinates, seg.parent.coordinates,
                                        seg.diameter, nslices)
            else: raise NotImplementedError
        vertices = [obj.get_vertices() for obj in objects]
        indices  = [obj.get_indices() for obj in objects]
        num_v    = np.cumsum([len(x) for x in vertices])
        for v_index, v_offset in zip(indices[1:], num_v):
            v_index += v_offset
        self.vertices = np.vstack(vertices)
        self.indices  = np.vstack(indices)
        self.segments = np.empty(len(self.vertices), dtype=Pointer)
        for idx in range(num_seg):
            lower = 0 if idx == 0 else num_v[idx-1]
            upper = num_v[idx]
            self.segments[lower:upper] = idx

        # TODO: Move these arrays to the GPU now, instead of copying them at
        # render time. Use VBO's?

    def draw(self, colors, interpolate=False):
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        self.draw_colors(colors)
        glDrawElements(GL_TRIANGLES, 3 * len(self.indices), GL_UNSIGNED_INT, self.indices)

    def draw_colors(self, colors):
        if colors is None: colors = [1, 1, 1]
        colors = np.array(colors, dtype=np.float32)
        if colors.shape == (3,) or colors.shape == (4,):
            colors = np.tile(colors, [len(self.vertices), 1])
        else:
            assert len(colors) == self.num_seg, "Model changed, but 3d mesh did not!"
            colors = np.take(colors, self.segments, axis=0)

        # TODO: assert all colors in range [0, 1]

        glEnableClientState(GL_COLOR_ARRAY)
        color_depth = colors.shape[1]
        if color_depth == 3:
            glColorPointer(3, GL_FLOAT, 0, colors)
        elif color_depth == 4:
            glColorPointer(4, GL_FLOAT, 0, colors)
        else: raise ValueError(color_depth)

class Viewport:
    def __init__(self, window_size=(2*640,2*480),
                move_speed = .02,
                mouse_sensitivity = .001,
                camera_position=[0.0, 0.0, 0.0],
                fps=60):
        pygame.init()
        pygame.display.set_mode(window_size, DOUBLEBUF|OPENGL)
        self.clock = pygame.time.Clock()

        self.window_size = window_size = pygame.display.get_window_size()
        self.window_center = [0.5 * x for x in window_size]
        self.background_color = [0,0,0,0]
        self.fps = float(fps)
        self.fov = 45.
        self.move_speed = float(move_speed)
        self.turn_speed = float(mouse_sensitivity)
        self.camera_pos   = np.array(camera_position, dtype=float)
        self.camera_pitch = 0.0
        self.camera_yaw   = 0.0
        self.update_camera_rotation_matrix()
        self.camera_forward = np.array([ 0.0, 0.0, -1.0])
        self.camera_up      = np.array([ 0.0, 1.0, 0.0])

        pygame.event.set_grab(True)

        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_FLAT) # Or "GL_SMOOTH"
        glDisable(GL_CULL_FACE)

    def set_scene(self, scene_or_database, *args):
        if isinstance(scene_or_database, Scene):
            self.scene = scene_or_database
        else:
            self.scene = Scene(scene_or_database, *args)

    def tick(self, colors=None):
        dt = self.clock.tick(self.fps)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.read_keyboard(dt)
        self.read_mouse(dt)
        self.setup_camera()
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        glClearColor(*self.background_color)
        self.scene.draw(colors=colors)
        pygame.display.flip()
        if False:
            print("Camera Position", self.camera_pos)
            print("Camera Pitch", self.camera_pitch)
            print("Camera Yaw  ", self.camera_yaw)

    def read_keyboard(self, dt):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            self.move_camera([0.0, 0.0, -self.move_speed * dt])
        if keys[pygame.K_s]:
            self.move_camera([0.0, 0.0, +self.move_speed * dt])
        if keys[pygame.K_a]:
            self.move_camera([-self.move_speed * dt, 0, 0])
        if keys[pygame.K_d]:
            self.move_camera([+self.move_speed * dt, 0, 0])
        if keys[pygame.K_SPACE]:
            self.camera_pos[1] += self.move_speed * dt
        if keys[pygame.K_LCTRL]:
            self.camera_pos[1] -= self.move_speed * dt

    def read_mouse(self, dt):
        if not pygame.event.get_grab():
            pygame.mouse.set_visible(True)
            return
        else:
            pygame.mouse.set_visible(False)
        x, y = pygame.mouse.get_rel()
        self.camera_yaw   += x * self.turn_speed
        self.camera_pitch += y * self.turn_speed
        halfpi = 0.5 * np.pi - 5000*epsilon
        self.camera_yaw = self.camera_yaw % (2.0 * np.pi)
        self.camera_pitch = np.clip(self.camera_pitch, -halfpi, +halfpi)
        self.update_camera_rotation_matrix()

    def update_camera_rotation_matrix(self):
        self.camera_rotation = (
                Rotation.from_euler('x', (self.camera_pitch)) *
                Rotation.from_euler('y', (self.camera_yaw))
        ).as_matrix()

    def move_camera(self, offset):
        """
        Move the camera position.
        The offset is relative to the camera's viewpoint, not the world.
        """
        self.camera_pos += np.array(offset).dot(self.camera_rotation)

    def setup_camera(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity();
        max_view_dist = 10e3
        gluPerspective(self.fov, (self.window_size[0]/self.window_size[1]), 0.1, max_view_dist)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        camera_lookat = self.camera_forward.dot(self.camera_rotation) + self.camera_pos
        gluLookAt(*self.camera_pos, *camera_lookat, *self.camera_up)
