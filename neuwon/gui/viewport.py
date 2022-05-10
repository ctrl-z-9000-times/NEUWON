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
        lod = float(lod)
        if hasattr(database, "get_database"): database = database.get_database()
        Segment = database.get("Segment")
        self.num_seg = num_seg = len(Segment)
        assert num_seg > 0
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
        assert np.all(colors >= 0.0)
        assert np.all(colors <= 1.0)
        if colors.shape == (3,) or colors.shape == (4,):
            colors = np.tile(colors, [len(self.vertices), 1])
        else:
            assert len(colors) == self.num_seg, "Model changed, but 3d mesh did not!"
            colors = np.take(colors, self.segments, axis=0)

        glEnableClientState(GL_COLOR_ARRAY)
        color_depth = colors.shape[1]
        if color_depth == 3:
            glColorPointer(3, GL_FLOAT, 0, colors)
        elif color_depth == 4:
            glColorPointer(4, GL_FLOAT, 0, colors)
        else: raise ValueError(color_depth)

    def get_segment(self, window_size, screen_coordinates):
        x, y = screen_coordinates
        y = window_size[1] - y

        colors = np.empty((len(self.vertices), 3), dtype=np.uint8)
        mask = 2**8 - 1
        np.bitwise_and(self.segments, mask,       out=colors[:,0])
        np.bitwise_and(self.segments, mask << 8,  out=colors[:,1])
        np.bitwise_and(self.segments, mask << 16, out=colors[:,2])

        # Only render a single pixel.
        glScissor(x, y, 1, 1)
        glEnable(GL_SCISSOR_TEST)

        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_UNSIGNED_BYTE, 0, colors)
        glDrawElements(GL_TRIANGLES, 3 * len(self.indices), GL_UNSIGNED_INT, self.indices)

        glFlush()
        glFinish()
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        color = glReadPixels(x, y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
        segment = color[0] + (color[1] << 8) + (color[2] << 16)
        print(segment)

        # Restore OpenGL settings.
        glDisable(GL_SCISSOR_TEST)


        # return segment

class Viewport:
    def __init__(self, window_size=(2*640,2*480),
                move_speed = .02,
                mouse_sensitivity = .001,):
        self.move_speed = float(move_speed)
        self.turn_speed = float(mouse_sensitivity)
        self.sprint_modifier = 5 # Shift key move_speed multiplier.
        # Camera settings.
        self.background_color = [0,0,0,0]
        self.fov = 45.
        self.max_view_dist  = 10e3
        self.camera_pos     = np.zeros(3)
        self.camera_pitch   = 0.0
        self.camera_yaw     = 0.0
        self.camera_up      = np.array([ 0.0, 1.0, 0.0]) # +Y is up.
        self.camera_forward = np.array([ 0.0, 0.0, -1.0])
        self.update_camera_rotation_matrix()
        # Setup pygame and OpenGL.
        self.alive = True
        pygame.init()
        pygame.display.set_mode(window_size, DOUBLEBUF|OPENGL)
        self.clock = pygame.time.Clock()
        self.window_size = pygame.display.get_window_size()
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_FLAT) # Or "GL_SMOOTH"
        glDisable(GL_CULL_FACE)

    def set_scene(self, scene_or_database, *args):
        if isinstance(scene_or_database, Scene):
            self.scene = scene_or_database
        else:
            self.scene = Scene(scene_or_database, *args)

    def close(self):
        pygame.quit()
        self.alive = False

    def tick(self, colors=None):
        dt = self.clock.tick()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self.mouse_pos = event.pos

        if pygame.mouse.get_focused():
            m1, m2, m3 = pygame.mouse.get_pressed()
            if m1:
                pygame.mouse.set_visible(False)
                self.read_keyboard(dt)
                self.read_mouse(dt)
            else:
                pygame.mouse.set_visible(True)
        else:
            pygame.mouse.set_visible(True)

        self.setup_camera()
        self.scene.get_segment(self.window_size, pygame.mouse.get_pos()) # DEBUGGING!

        self.setup_camera()
        glClearColor(*self.background_color)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        self.scene.draw(colors=colors)
        pygame.display.flip()
        if False:
            print("Camera Position", self.camera_pos)
            print("Camera Pitch", self.camera_pitch)
            print("Camera Yaw  ", self.camera_yaw)

    def read_keyboard(self, dt):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            dt *= self.sprint_modifier
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.move_camera([0.0, 0.0, -self.move_speed * dt])
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.move_camera([0.0, 0.0, +self.move_speed * dt])
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.move_camera([-self.move_speed * dt, 0, 0])
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.move_camera([+self.move_speed * dt, 0, 0])
        if keys[pygame.K_SPACE] or keys[pygame.K_RETURN]:
            self.camera_pos[1] += self.move_speed * dt
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            self.camera_pos[1] -= self.move_speed * dt

    def read_mouse(self, dt):
        # Get the relative movement of the mouse cursor.
        x,  y  = pygame.mouse.get_pos()
        x0, y0 = self.mouse_pos
        dx = x - x0
        dy = y - y0
        pygame.mouse.set_pos(self.mouse_pos)
        # Apply mouse movement.
        self.camera_yaw   += dx * self.turn_speed
        self.camera_pitch += dy * self.turn_speed
        # Keep camera angle in sane mathematical bounds.
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
        gluPerspective(self.fov, (self.window_size[0]/self.window_size[1]), 0.1, self.max_view_dist)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        camera_lookat = self.camera_forward.dot(self.camera_rotation) + self.camera_pos
        gluLookAt(*self.camera_pos, *camera_lookat, *self.camera_up)
