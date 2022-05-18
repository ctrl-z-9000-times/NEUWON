import numpy as np
import pygame
from pygame.locals import *
from OpenGL import GL
from .coloration import Coloration
from .text_overlay import TextOverlay
from .scene import Camera, Scene
import enum
import collections.abc
import queue

epsilon = np.finfo(float).eps

class Message(enum.Enum):
    OPEN            = enum.auto()
    SET_SCENE       = enum.auto()
    SET_COLORMAP    = enum.auto()
    SET_VALUES      = enum.auto()
    SET_VISIBLE     = enum.auto()
    SET_BACKGROUND  = enum.auto()
    SHOW_TEXT       = enum.auto()
    SHOW_TIME       = enum.auto()
    SHOW_TYPE       = enum.auto()
    QUIT            = enum.auto()

class _Clock:
    def __init__(self, fps):
        self._period    = 1 / fps
        self._prev_tick = time.time()

    def tick(self):
        t       = time.time()
        dt      = t - self._prev_tick
        delay   = self._period - dt
        if delay > 0:
            time.sleep(delay)
            t  = time.time()
            dt = t - self._prev_tick
        self._prev_tick = t
        return dt

class Viewport:
    """
    This class opens the viewport window, embeds the rendered scene,
    and handles the user input.
    """
    def __init__(self,
                move_speed = .02,
                mouse_sensitivity = .001,
                sprint_modifier = 5):
        self.move_speed     = float(move_speed)
        self.turn_speed     = float(mouse_sensitivity)
        self.sprint_mod     = float(sprint_modifier) # Shift key move_speed multiplier.
        self.control_queue  = queue.Queue()
        self._coloration    = Coloration()
        self._text_overlay  = TextOverlay()
        self._is_open       = False
        self._scene         = None

    def _update_control(self):
        while True:
            try:
                m = self.control_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(m, collections.abc.Iterable):
                m, payload = m
            assert isinstance(m, Message)

            if   m == Message.OPEN:         self._open(payload)
            elif m == Message.SET_SCENE:    self._set_scene(payload)
            elif m == Message.SET_COLORMAP: self._coloration.set_colormap(payload)
            elif m == Message.SET_VALUES:   self._coloration.set_segment_values(payload)
            elif m == Message.SET_VISIBLE:  self._coloration.set_visible_segments(payload)
            elif m == Message.SET_BACKGROUND: self._coloration.set_background_color(payload)
            elif m == Message.SHOW_TEXT:    self._text_overlay.show_text(payload)
            elif m == Message.SHOW_TIME:    self._text_overlay.show_time(payload)
            elif m == Message.SHOW_TYPE:    self._text_overlay.show_type(payload)
            elif m == Message.QUIT:         self._close()
            else: raise NotImplementedError(m)

    def _open(self, window_size):
        if self._is_open:
            return
        self._left_click = False
        self._is_open       = True
        # Setup pygame.
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(window_size, OPENGL)
        self.font   = pygame.font.SysFont(None, 24)
        self.clock  = pygame.time.Clock()
        self.camera = Camera(pygame.display.get_window_size(), 45, 10e3)

    def _close(self):
        if not self._is_open:
            return
        pygame.mouse.set_visible(True)
        pygame.display.quit()
        pygame.quit()
        self._is_open   = False
        self._scene     = None
        self._coloration.clear_data()

    def is_open(self):
        return self._is_open

    def _set_scene(self, model):
        self._scene = Scene(model)
        self._coloration.set_segment_values(np.zeros(self._scene.num_seg))
        self._coloration.set_visible_segments(np.arange(self._scene.num_seg))

    def tick(self):
        self._update_control()
        if not self._is_open:
            return
        dt = self.clock.tick()
        # Process queued events.
        right_click = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._close()
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
        if right_click or self._text_overlay._show_type:
            segment = self._scene.get_segment(self.camera, self._coloration, pygame.mouse.get_pos())
        else:
            segment = None
        # 
        self._scene.draw(self.camera, self._coloration)
        overlay = self._text_overlay._get(segment)
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
        r,g,b   = self._coloration.background_color
        color   = (255 * (1 - r), 255 * (1 - g), 255 * (1 - b), 255)
        x, y    = position
        for line in text.split('\n'):
            overlay = self.font.render(line, True, color).convert_alpha()
            width   = overlay.get_width()
            height  = overlay.get_height()
            data    = pygame.image.tostring(overlay, "RGBA", True)
            GL.glWindowPos2d(x, y)
            GL.glDrawPixels(width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
            y -= round(height * 1.10)
