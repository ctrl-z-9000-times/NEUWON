import math
import numpy as np
import pygame
from pygame.locals import *
from scipy.spatial.transform import Rotation

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from neuwon.database import Database, Pointer, epsilon

__all__ = ["Scene", "Viewport"]

# MEMO: Don't optimize this code.

class Scene:
    def __init__(self, database, lod=2.5):
        lod     = float(lod)
        Segment = database.get("Segment")
        objects = np.zeros(len(Segment), dtype=object)
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
        offsets  = np.cumsum([len(x) for x in vertices])
        self.segments = np.empty(offsets[-1], dtype=Pointer)
        for idx in range(len(Segment)):
            lower = 0 if idx == 0 else offsets[idx-1]
            upper = offsets[idx]
            indices[idx] += lower
            self.segments[lower:upper] = idx
        self.vertices = np.vstack(vertices)
        self.indices  = np.vstack(indices)
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

class Primative:
    def get_vertices(self):
        return self.vertices
    def get_indices(self):
        return self.indices

class Sphere(Primative):
    def __init__(self, center, radius, slices):
        stacks = slices
        # Calculate the Vertices
        num_vertices = (stacks + 1) * (slices + 1)
        vertices = np.zeros((num_vertices, 3), dtype=np.float32)
        write_idx = 0
        for i in range(stacks + 1):
            phi = math.pi * (i / stacks)
            # Loop Through Slices
            for j in range(slices + 1):
                theta = 2 * math.pi * (j / slices)
                # Calculate The Vertex Positions
                vertices[write_idx][0] = math.cos(theta) * math.sin(phi)
                vertices[write_idx][1] = math.cos(phi)
                vertices[write_idx][2] = math.sin(theta) * math.sin(phi)
                write_idx += 1
        vertices *= radius
        vertices += center
        # Calculate The Index Positions
        num_triangles = 2 * slices * (stacks + 1)
        indices = np.zeros((num_triangles, 3), dtype=np.int32)
        write_idx = 0
        for i in range(slices * stacks + slices):
            indices[write_idx, 0] = i
            indices[write_idx, 1] = i + slices + 1
            indices[write_idx, 2] = i + slices
            write_idx += 1
            indices[write_idx, 0] = i + slices + 1
            indices[write_idx, 1] = i
            indices[write_idx, 2] = i + 1
            write_idx += 1
        self.vertices = vertices
        self.indices  = indices

class Cylinder(Primative):
    def __init__(self, A, B, diameter, num_slices):
        triangle_strip = np.empty((2 * (num_slices + 1), 3), dtype=np.float32)
        vector = B - A
        length = np.linalg.norm(vector)
        vector /= length
        ref_vector = np.array([0.0, 0.0, 1.0])
        rot_matrix = Rotation.align_vectors(ref_vector.reshape((-1,3)), vector.reshape((-1,3)))[0].as_matrix()
        for s in range(num_slices + 1):
            f = 2 * math.pi * (s / num_slices)
            y = math.sin(f)
            x = math.cos(f)
            triangle_strip[2*s,   0] = x
            triangle_strip[2*s,   1] = y
            triangle_strip[2*s,   2] = 0.
            triangle_strip[2*s+1, 0] = x
            triangle_strip[2*s+1, 1] = y
            triangle_strip[2*s+1, 2] = length
        triangle_strip = triangle_strip.dot(rot_matrix)
        triangle_strip += A
        self.vertices = triangle_strip
        self.indices  = np.empty((2 * num_slices, 3), dtype=np.uint32)
        for i in range(num_slices):
            self.indices[2*i,   2] = i * 2
            self.indices[2*i,   1] = i * 2 + 1
            self.indices[2*i,   0] = i * 2 + 2
            self.indices[2*i+1, 2] = i * 2 + 3
            self.indices[2*i+1, 1] = i * 2 + 2
            self.indices[2*i+1, 0] = i * 2 + 1

class Disk:
    def __init__(self,):
        1/0 # todo maybe?
        triangle_fan = np.empty((num_slices, 3), dtype=np.float32)
        for s in range(num_slices):
            f = 2 * math.pi * (s / num_slices)
            y = math.sin(f)
            x = math.cos(f)
            triangle_fan[s, 0] = x
            triangle_fan[s, 1] = y
            triangle_fan[s, 2] = 0.
        triangle_fan = triangle_fan.dot(rot_matrix)
        triangle_fan += A

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
        if isinstance(scene_or_database, Database):
            self.scene = Scene(scene_or_database, *args)
        elif isinstance(scene_or_database, Scene):
            self.scene = scene_or_database
        else: raise TypeError(scene_or_database)

    def tick(self):
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
        self.scene.draw()
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
