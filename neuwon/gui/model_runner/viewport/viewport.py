import numpy as np
import pygame
from pygame.locals import *
from OpenGL import GL
from .coloration import Coloration
from .text_overlay import TextOverlay
from .scene import Camera, Scene
import enum
import collections.abc
from multiprocessing import Process, Pipe

epsilon = np.finfo(float).eps

class ViewportAPI:
    def __init__(self, window_size):
        viewport = ViewportImpl(window_size)
        self._pipe, other_end = Pipe()
        self._process = Process(target=viewport.mainloop, args=(other_end,),
                                name="Viewport")
        self._process.start()

    def _send(self, message, *args):
        assert isinstance(message, Message)
        try:
            self._pipe.send((message, args))
        except BrokenPipeError:
            self._process.join()

    def __del__(self):
        self._send(Message.CLOSE)

    def set_model(self, model):
        database = model.get_database()
        assert database.is_sorted()
        parent      = database.get_data('Segment.parent')
        coordinates = database.get_data('Segment.coordinates')
        diameter    = database.get_data('Segment.diameter')
        self._send(Message.SET_MODEL, parent, coordinates, diameter)

    def set_colors(self, colors):
        self._send(Message.SET_COLORS, colors)

    def set_visible(self, visible):
        self._send(Message.SET_VISIBLE, visible)

    def set_background(self, color):
        self._send(Message.SET_BACKGND, color)

    def set_text(self, text):
        self._send(Message.SET_TEXT, text)

    def get_selected(self) -> 'Segment-Index':
        return None
        if self._process.stdout.poll():
            segment_index = self.process.stdout.recv()
            segment_index = int(segment_index)
            return segment_index

class Message(enum.Enum):
    CLOSE           = enum.auto()
    SET_MODEL       = enum.auto()
    SET_COLORS      = enum.auto()
    SET_VISIBLE     = enum.auto()
    SET_BACKGND     = enum.auto()
    SET_TEXT        = enum.auto()

class ViewportImpl:
    def __init__(self, window_size,
                move_speed = .02,
                mouse_sensitivity = .001,
                sprint_modifier = 5):
        self.window_size    = window_size
        self.move_speed     = float(move_speed)
        self.turn_speed     = float(mouse_sensitivity)
        self.sprint_mod     = float(sprint_modifier) # Shift key move_speed multiplier.
        self._is_open       = False
        self._scene         = None
        self._coloration    = Coloration()
        self._text_overlay  = TextOverlay()

    def mainloop(self, pipe):
        self._pipe = pipe
        self._open()
        while self._is_open:
            self._tick()
            self._recv_messages()

    def _open(self):
        self._is_open = True
        self._left_click = False
        # Setup pygame.
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(self.window_size, OPENGL)
        self.font   = pygame.font.SysFont(None, 24)
        self.clock  = pygame.time.Clock()
        self.camera = Camera(pygame.display.get_window_size(), 45, 10e3)

    def _close(self):
        pygame.mouse.set_visible(True)
        pygame.display.quit()
        pygame.quit()
        self._is_open = False

    def _recv_messages(self):
        while self._pipe.poll():
            m, args = self._pipe.recv()
            if   m == Message.SET_MODEL:    self._set_model(*args)
            elif m == Message.SET_COLORS:   self._coloration.set_colormap(*args)
            elif m == Message.SET_VISIBLE:  self._coloration.set_visible_segments(*args)
            elif m == Message.SET_BACKGND:  self._coloration.set_background_color(*args)
            elif m == Message.SET_TEXT:     self._text_overlay.show_text(*args)
            elif m == Message.CLOSE:        self._close()
            else: raise NotImplementedError(m)

    def _set_model(self, *args):
        self._scene = Scene(*args)
        self._coloration.set_segment_values(np.zeros(self._scene.num_seg))
        self._coloration.set_visible_segments(np.arange(self._scene.num_seg))

    def _tick(self):
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
        if self._scene and (right_click or self._text_overlay._show_type):
            segment = self._scene.get_segment(self.camera, self._coloration, pygame.mouse.get_pos())
        else:
            segment = None
        # 
        if self._scene:
            self._scene.draw(self.camera, self._coloration)
        overlay = self._text_overlay._get(segment)
        self._draw_text_overlay(overlay, (40, 50))
        pygame.display.flip()
        # 
        if right_click:
            print(segment)

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
