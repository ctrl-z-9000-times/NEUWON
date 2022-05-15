import matplotlib.pyplot as plt
import numpy as np
import pygame
from pygame.locals import *
from OpenGL import GL
from .scene import Camera, Scene

epsilon = np.finfo(float).eps

class TextOverlay:
    """ Holds the state of the standard text overlay. """
    def __init__(self):
        self.set_text('')
        self.set_neuron_type(True)
        self.set_segment_type(True)

    def set_text(self, text):
        self.text = str(text)

    def set_neuron_type(self, value):
        self.neuron_type = bool(value)

    def set_segment_type(self, value):
        self.segment_type = bool(value)

    def _need_segment(self):
        return self.neuron_type or self.segment_type

    def _get(self, segment=None):
        overlay = self.text
        if segment is None:
            neuron_type  = None
            segment_type = None
        else:
            neuron_type  = segment.neuron.neuron_type
            segment_type = segment.segment_type
        if self.neuron_type:  overlay += f'\nNeuron Type: {neuron_type}'
        if self.segment_type: overlay += f'\nSegment Type: {segment_type}'
        return overlay.strip()

class Coloration:
    """ Holds the segments color & visibility data, manages colormaps. """
    def __init__(self):
        self.segment_values     = None
        self.visible_segments   = None
        self.color_data         = None
        self.set_colormap(self.get_all_colormaps()[0])

    @classmethod
    def get_all_colormaps(cls):
        return plt.colormaps()

    def set_colormap(self, colormap):
        self.colormap = plt.get_cmap(colormap)

    def set_segment_values(self, segment_values):
        self.segment_values = segment_values

    def set_visible_segments(self, visible_segments):
        self.visible_segments = visible_segments

    def _get(self):
        if self.segment_values is None:
            return None
        return self.colormap(self.segment_values, self.visible_segments)

class Viewport:
    """
    This class opens the viewport window, embeds the rendered scene,
    and handles the user input.
    """
    def __init__(self, window_size=(2*640,2*480),
                move_speed = .02,
                mouse_sensitivity = .001,):
        self.move_speed     = float(move_speed)
        self.turn_speed     = float(mouse_sensitivity)
        self.sprint_mod     = 5 # Shift key move_speed multiplier.
        self.coloration     = Coloration()
        self.text_overlay   = TextOverlay()
        self.set_background_color('black')
        self._open = False
        self.open(window_size)

    def open(self, window_size):
        if self._open:
            return
        self._left_click = False
        self._open       = True
        # Setup pygame.
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(window_size, OPENGL)
        self.font   = pygame.font.SysFont(None, 24)
        self.clock  = pygame.time.Clock()
        self.camera = Camera(pygame.display.get_window_size(), 45, 10e3)

    def close(self):
        if not self._open:
            return
        pygame.mouse.set_visible(True)
        pygame.display.quit()
        pygame.quit()
        self._open = False
        # Drop the big data buffers.
        del self._scene
        self.coloration.segment_values = None

    def is_open(self):
        return self._open

    def set_scene(self, model):
        self._scene = Scene(model)

    def set_background_color(self, color):
        if isinstance(color, str):
            color = color.lower()
            if color == 'black':
                self.background_color = [0,0,0,0]
            elif color == 'white':
                self.background_color = [1,1,1,1]
            else:
                raise NotImplementedError(color)
        else:
            self.background_color = list(float(x) for x in color)

    def get_coloration(self):
        return self.coloration

    def get_text_overlay(self):
        return self.text_overlay

    def tick(self):
        dt = self.clock.tick()
        # Process queued events.
        right_click = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._left_click     = True
                    self._left_click_pos = event.pos
                    pygame.mouse.set_visible(False)
                elif event.button == 3:
                    right_click = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self._left_click = False
                    pygame.mouse.set_visible(True)
            # Restore the mouse cursor if it leaves the window.
            elif event.type in (pygame.WINDOWFOCUSLOST, pygame.WINDOWLEAVE,
                                pygame.WINDOWCLOSE,     pygame.WINDOWMINIMIZED):
                self._left_click = False
                pygame.mouse.set_visible(True)
        # Holding left mouse button triggers camera movement.
        if self._left_click:
            self._mouse_movement(dt)
            self._keyboard_movement(dt)
        # Press right mouse button to select a segment.
        if right_click or self.text_overlay._need_segment():
            segment = self._scene.get_segment(self.camera, pygame.mouse.get_pos())
        else:
            segment = None
        # 
        self._scene.draw(self.camera, self.coloration._get(), self.background_color)
        overlay = self.text_overlay._get(segment)
        self._draw_text_overlay(overlay, (40, 50))
        pygame.display.flip()
        # 
        if right_click:
            return segment

    def _mouse_movement(self, dt):
        # Get the relative movement of the mouse cursor.
        x,  y  = pygame.mouse.get_pos()
        x0, y0 = self._left_click_pos
        dx = x - x0
        dy = y - y0
        pygame.mouse.set_pos(self._left_click_pos)
        # Apply mouse movement.
        self.camera.yaw   += dx * self.turn_speed
        self.camera.pitch += dy * self.turn_speed
        # Keep camera angles in sane mathematical bounds.
        halfpi = 0.5 * np.pi - 5000*epsilon
        self.camera.yaw = self.camera.yaw % (2.0 * np.pi)
        self.camera.pitch = np.clip(self.camera.pitch, -halfpi, +halfpi)
        self.camera.update_rotation_matrix()

    def _keyboard_movement(self, dt):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            dt *= self.sprint_mod
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

    def _draw_text_overlay(self, text, position):
        if not text:
            return
        r,g,b,a = self.background_color
        color   = (255 - r, 255 - g, 255 - b, 255)
        x, y    = position
        for line in text.split('\n'):
            overlay = self.font.render(line, True, color).convert_alpha()
            width   = overlay.get_width()
            height  = overlay.get_height()
            data    = pygame.image.tostring(overlay, "RGBA", True)
            GL.glWindowPos2d(x, y)
            GL.glDrawPixels(width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
            y -= round(height * 1.10)
