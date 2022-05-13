import numpy as np
import pygame
from pygame.locals import *
from OpenGL import GL
from .scene import Camera, Scene

epsilon = np.finfo(float).eps

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
        self.camera = Camera(pygame.display.get_window_size(), 45, 10e3)

    def set_scene(self, database):
        self.scene = Scene(database)

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
            GL.glWindowPos2d(x, y)
            GL.glDrawPixels(width, height, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data)
            y -= round(height * 1.10)
