import math
import numpy as np
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

from neuwon.database import Pointer

# MEMO: Don't optimize this code.

class Scene:
    def __init__(self, database, lod=.5):
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
        ref_vector = np.array([0, 0, 1], dtype=np.float32)
        rot_matrix = _rotateAlign(ref_vector, vector)
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
            self.indices[2*i,   0] = i * 2
            self.indices[2*i,   1] = i * 2 + 1
            self.indices[2*i,   2] = i * 2 + 2
            self.indices[2*i+1, 0] = i * 2 + 3
            self.indices[2*i+1, 1] = i * 2 + 2
            self.indices[2*i+1, 2] = i * 2 + 1

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

def _rotateAlign(v1, v2):
    # https://gist.github.com/kevinmoran/b45980723e53edeb8a5a43c49f134724
    # https://iquilezles.org/www/articles/noacos/noacos.htm
    axis = np.cross(v1, v2)
    axis_x = axis[0]
    axis_y = axis[1]
    axis_z = axis[2]
    cosA = np.dot(v1, v2)
    k = 1. / (1. + cosA)
    result = np.array([
            (axis_x * axis_x * k) + cosA,
            (axis_y * axis_x * k) - axis_z,
            (axis_z * axis_x * k) + axis_y,
            (axis_x * axis_y * k) + axis_z,
            (axis_y * axis_y * k) + cosA,
            (axis_z * axis_y * k) - axis_x,
            (axis_x * axis_z * k) - axis_y,
            (axis_y * axis_z * k) + axis_x,
            (axis_z * axis_z * k) + cosA
    ], dtype=np.float32).reshape((3,3))
    return result


def main():

    from neuwon.database import Database
    from neuwon.segment import SegmentMethods

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

    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 500.0)

    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_FLAT) # Or "GL_SMOOTH"
    glDisable(GL_CULL_FACE)
    # glEnable(GL_CULL_FACE)

    glTranslatef(0.0,0.0, -5)

    db = ball_and_stick()

    move_speed = .5
    move_direction = (0.0,0.0,0.0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        move_direction = [0.0,0.0,0.0]
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            move_direction[2] += move_speed
        if keys[pygame.K_a]:
            move_direction[0] += move_speed
        if keys[pygame.K_s]:
            move_direction[2] -= move_speed
        if keys[pygame.K_d]:
            move_direction[0] -= move_speed
        if keys[pygame.K_SPACE]:
            move_direction[1] -= move_speed
        if keys[pygame.K_LCTRL]:
            move_direction[1] += move_speed
        glTranslatef(*move_direction)

        # glRotatef(1, 3, 1, 1)
        glClearColor(0,0,0,0) # Background color.
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        scene = Scene(db)
        scene.draw(None)

        pygame.display.flip()
        pygame.time.wait(10)

if __name__ == "__main__": main()
