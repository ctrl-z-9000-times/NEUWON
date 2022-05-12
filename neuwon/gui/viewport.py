import numpy as np
import pygame
from pygame.locals import *
from scipy.spatial.transform import Rotation

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from neuwon.database import Pointer
from neuwon.gui.primatives import Sphere, Cylinder

epsilon = np.finfo(float).eps

class _Camera:
    def __init__(self, size, fov, dist):
        self.size    = size # Window size
        self.fov     = fov  # Field of view
        self.dist    = dist # Maximum view distance
        self.pos     = np.zeros(3)
        self.pitch   = 0.0
        self.yaw     = 0.0
        self.up      = np.array([ 0.0, 1.0, 0.0]) # +Y is up.
        self.forward = np.array([ 0.0, 0.0, -1.0])
        self.update_rotation_matrix()

    def update_rotation_matrix(self):
        self.rotation = (
                Rotation.from_euler('x', (self.pitch)) *
                Rotation.from_euler('y', (self.yaw))
        ).as_matrix()

    def move(self, offset):
        """
        Move the camera position.
        The offset is relative to the camera's viewpoint, not the world.
        """
        self.pos += np.array(offset).dot(self.rotation)

    def setup_opengl(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity();
        gluPerspective(self.fov, (self.size[0]/self.size[1]), 0.1, self.dist)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        lookat = self.forward.dot(self.rotation) + self.pos
        gluLookAt(*self.pos, *lookat, *self.up)

    def print(self):
        print("Camera Position", self.pos)
        print("Camera Pitch",    self.pitch)
        print("Camera Yaw  ",    self.yaw)

class _Scene:
    """ This class generates the 3D mesh and renders it using OpenGL. """
    def __init__(self, database, lod=2.5):
        self._init_opengl_settings()
        if hasattr(database, "get_database"):
            database = database.get_database()
        assert database.is_sorted()
        Segment = database.get("Segment")
        self.num_seg = num_seg = len(Segment)
        if num_seg == 0:
            self.vertices = np.empty(0)
            self.indices  = np.empty(0)
            self.segments = np.empty(0)
            return
        # Collect all of the 3D objects, one per segment.
        objects = np.zeros(num_seg, dtype=object)
        for idx in range(num_seg):
            seg = Segment.index_to_object(idx)
            nslices = max(3, int(lod * seg.diameter))
            if seg.is_sphere():
                objects[idx] = Sphere(seg.coordinates, 0.5 * seg.diameter, nslices)
            elif seg.is_cylinder():
                objects[idx] = Cylinder(seg.coordinates, seg.parent.coordinates,
                                        seg.diameter, nslices)
            else: raise NotImplementedError
        # Combine all of the objects into one big object.
        vertices = [obj.get_vertices() for obj in objects]
        indices  = [obj.get_indices() for obj in objects]
        num_v    = np.cumsum([len(x) for x in vertices])
        for v_index, v_offset in zip(indices[1:], num_v):
            v_index += v_offset
        self.vertices = np.vstack(vertices)
        self.indices  = np.vstack(indices)
        # Record which segment generated each vertex.
        self.segments = np.empty(len(self.vertices), dtype=Pointer)
        for idx in range(num_seg):
            lower = 0 if idx == 0 else num_v[idx-1]
            upper = num_v[idx]
            self.segments[lower:upper] = idx

        # TODO: Move these arrays to the GPU now, instead of copying them at
        # render time. Use VBO's?

    def _init_opengl_settings(self):
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_FLAT) # Or "GL_SMOOTH"
        glDisable(GL_CULL_FACE)
        # Setup for overlaying text with a transparent background.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def draw(self, camera, colors=None, background_color=None):
        camera.setup_opengl()
        colors = self._clean_color_data(colors)
        self._draw_background(background_color)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)

        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(colors.shape[1], GL_FLOAT, 0, colors)

        glDrawElements(GL_TRIANGLES, 3 * len(self.indices), GL_UNSIGNED_INT, self.indices)

    def _clean_color_data(self, colors):
        """ Clean up the user input and broadcast it to the correct shape. """
        if colors is None: colors = [1, 1, 1]
        colors = np.array(colors, dtype=np.float32)
        assert np.all(colors >= 0.0)
        assert np.all(colors <= 1.0)
        if colors.shape == (3,) or colors.shape == (4,):
            colors = np.tile(colors, [len(self.vertices), 1])
        else:
            assert len(colors) == self.num_seg, "Model changed, but 3d mesh did not!"
            colors = np.take(colors, self.segments, axis=0)
        assert colors.shape[1] in (3, 4)
        return colors

    def _draw_background(self, background_color):
        if background_color is None:
            background_color = [0,0,0,0]
        assert len(background_color) == 4
        glClearColor(*background_color)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    def get_segment(self, camera, screen_coordinates):
        camera.setup_opengl()
        x, y = screen_coordinates
        y = camera.size[1] - y

        background = 2**24 - 1
        assert self.num_seg < background

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

        # Restore OpenGL settings.
        glDisable(GL_SCISSOR_TEST)

        if segment == background:
            return None
        else:
            return segment

class Viewport:
    """
    This class opens the viewport window, embeds the rendered scene,
    and handles the user input.
    """
    def __init__(self, window_size=(2*640,2*480),
                move_speed = .02,
                mouse_sensitivity = .001,):
        self.move_speed = float(move_speed)
        self.turn_speed = float(mouse_sensitivity)
        self.sprint_modifier = 5 # Shift key move_speed multiplier.
        self.background_color = [0,0,0,0]
        self.alive = True
        # Setup pygame.
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(window_size, OPENGL)
        self.font   = pygame.font.SysFont(None, 24)
        self.clock  = pygame.time.Clock()
        self.camera = _Camera(pygame.display.get_window_size(), 45, 10e3)

    def set_scene(self, database):
        self.scene = _Scene(database)

    def close(self):
        pygame.display.quit()
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
            if event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    import tkinter as tk
                    m = tk.Menu(tearoff=False)
                    m.add_command(label='Hello', command=lambda: print('Hello pygame!'))
                    m.tk_popup(event.pos[0], event.pos[1])

        if pygame.mouse.get_focused():
            m1, m2, m3 = pygame.mouse.get_pressed()
            if m1:
                pygame.mouse.set_visible(False)
                self._read_mouse(dt)
                self._read_keyboard(dt)
            else:
                pygame.mouse.set_visible(True)
        else:
            pygame.mouse.set_visible(True)

        seg_id = self.scene.get_segment(self.camera, pygame.mouse.get_pos())

        self.scene.draw(self.camera, colors, self.background_color)
        self._draw_overlay(f'segment {seg_id}\ncurrent time', (50,50))
        pygame.display.flip()

    def _read_mouse(self, dt):
        # Get the relative movement of the mouse cursor.
        x,  y  = pygame.mouse.get_pos()
        x0, y0 = self.mouse_pos
        dx = x - x0
        dy = y - y0
        pygame.mouse.set_pos(self.mouse_pos)
        # Apply mouse movement.
        self.camera.yaw   += dx * self.turn_speed
        self.camera.pitch += dy * self.turn_speed
        # Keep camera angles in sane mathematical bounds.
        halfpi = 0.5 * np.pi - 5000*epsilon
        self.camera.yaw = self.camera.yaw % (2.0 * np.pi)
        self.camera.pitch = np.clip(self.camera.pitch, -halfpi, +halfpi)
        self.camera.update_rotation_matrix()

    def _read_keyboard(self, dt):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            dt *= self.sprint_modifier
        if keys[pygame.K_w] or keys[pygame.K_UP]:
            self.camera.move([0.0, 0.0, -self.move_speed * dt])
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:
            self.camera.move([0.0, 0.0, +self.move_speed * dt])
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:
            self.camera.move([-self.move_speed * dt, 0, 0])
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
            self.camera.move([+self.move_speed * dt, 0, 0])
        if keys[pygame.K_SPACE] or keys[pygame.K_RETURN]:
            self.camera.pos[1] += self.move_speed * dt
        if keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]:
            self.camera.pos[1] -= self.move_speed * dt

    def _draw_overlay(self, text, position):
        r,g,b,a = self.background_color
        color   = (255 - r, 255 - g, 255 - b, 255)
        x, y    = position
        for line in text.split('\n'):
            overlay = self.font.render(line, True, color).convert_alpha()
            width   = overlay.get_width()
            height  = overlay.get_height()
            data    = pygame.image.tostring(overlay, "RGBA", True)
            glWindowPos2d(x, y)
            glDrawPixels(width, height, GL_RGBA, GL_UNSIGNED_BYTE, data)
            y -= round(height * 1.10)
