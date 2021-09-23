import math
import numpy as np
import pygame
from pygame.locals import *
from scipy.spatial.transform import Rotation

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from neuwon.database import Database, Pointer, epsilon
from neuwon.gui.primatives import *

__all__ = ["Scene", "Viewport"]

# MEMO: Don't optimize this code.

class Scene:
    def __init__(self, database, lod=2.5):
        lod     = float(lod)
        if hasattr(database, "get_database"): database = database.get_database()
        Segment = database.get("Segment")
        num_seg = len(Segment)
        objects = np.zeros(num_seg, dtype=object)
        for seg in Segment.get_all_instances():
            idx = seg.get_unstable_index()
            nslices = max(3, int(lod * seg.diameter))
            if seg.is_sphere():
                objects[idx] = Sphere(seg.coordinates, 0.5 * seg.diameter, nslices)
            elif seg.is_cylinder():
                objects[idx] = Cylinder(seg.coordinates, seg.parent.coordinates,
                                        seg.diameter, nslices)
        vertices = [obj.get_vertices() for obj in objects]
        indices  = [obj.get_indices() for obj in objects]
        self.vertices = np.vstack(vertices)
        self.indices  = np.vstack(indices)
        self.segments = np.empty(len(self.vertices), dtype=Pointer)
        lower = 0
        for idx in range(num_seg):
            upper = lower + len(vertices[idx])
            self.segments[lower:upper] = idx
            lower = upper
        # TODO: Move these arrays to the GPU now, instead of copying them at
        # render time. Use VBO's?

    def draw(self, colors=None, interpolate=False):
        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)

        if colors is None:
            colors = np.tile([1.,1.,1.], len(self.vertices))
        elif interpolate:
            1/0
        else:
            colors = colors[self.segments]
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, 0, colors)

        glDrawElements(GL_TRIANGLES, 3 * len(self.indices), GL_UNSIGNED_INT, self.indices);

class Viewport:
    def __init__(self, window_size=(2*640,2*480), move_speed = .02):
        pygame.init()
        pygame.display.set_mode(window_size, DOUBLEBUF|OPENGL)
        self.clock = pygame.time.Clock()

        self.window_size = window_size = pygame.display.get_window_size()
        self.background_color = [0,0,0,0]
        self.fps = 60.
        self.fov = 45.
        self.move_speed = float(move_speed)
        self.turn_speed = float(move_speed) / 100
        self.camera_pos   = np.array([0.0, 0.0, 0.0])
        self.camera_pitch = 0.0
        self.camera_yaw   = 0.0
        self.update_camera_rotation_matrix()
        self.camera_forward = np.array([ 0.0, 0.0, -1.0])
        self.camera_up      = np.array([ 0.0, 1.0, 0.0])

        pygame.mouse.set_visible(False)
        self.window_center = [0.5 * x for x in window_size]

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
        mouse_pos = pygame.mouse.get_pos()
        pygame.mouse.set_pos(self.window_center)
        delta = [mouse_pos[dim] - self.window_center[dim] for dim in range(2)]
        self.camera_yaw   += delta[0] * self.turn_speed * dt
        self.camera_pitch += delta[1] * self.turn_speed * dt
        halfpi = 0.5 * np.pi - 100*epsilon
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
        gluPerspective(self.fov, (self.window_size[0]/self.window_size[1]), 0.1, 10000.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        camera_lookat = self.camera_forward.dot(self.camera_rotation) + self.camera_pos
        gluLookAt(*self.camera_pos, *camera_lookat, *self.camera_up)

def main():

    from neuwon.segment import SegmentMethods
    import neuwon.regions
    import neuwon.growth

    def ball_and_stick():
        db = Database()
        SegmentMethods._initialize(db)
        Segment = db.get("Segment").get_instance_type()
        ball = Segment(parent=None, coordinates=[0,0,0], diameter=12)
        stick = []
        tip = ball
        for i in range(10):
            tip = Segment(parent=tip, coordinates=[i+6+1,0,0], diameter=3)
            stick.append(tip)
        return db

    def cell():
        db = Database()
        SegmentMethods._initialize(db)
        Segment = db.get("Segment").get_instance_type()
        region = neuwon.regions.Sphere([0,0,0], 100)
        soma = Segment(None, [0,0,0], 8)
        dendrites = neuwon.growth.Tree(soma, region, 0.0005,
                balancing_factor = .7,
                extension_distance = 40,
                bifurcation_distance = 40,
                extend_before_bifurcate = False,
                only_bifurcate = True,
                maximum_segment_length = 20,
                diameter = 1.5,
        )
        dendrites.grow()
        print("NUM SEGMENTS:", len(dendrites.get_segments()))
        return db

    db = cell()
    # db = ball_and_stick()

    view = Viewport()
    view.set_scene(db)

    while True:
        view.tick()


if __name__ == "__main__": main()
