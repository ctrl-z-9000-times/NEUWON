import math
import numpy as np
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import OpenGL.arrays.vbo as glvbo


# Opengl Renderer
#    -> great, cylinder works
#    -> now make a a simple neur0n and make it look pretty.
#        * Read straight from the database, see the segment class prototype
#    -> After the minimum viable prototype works, organize the code and see
#       about caching vertex data instead of recomputing it every frame.


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

class Cylinder:
    def __init__(self):
        self.tube = None
        self.cap  = None

    @staticmethod
    def calculate_vertexes(A, B, diameter, num_slices=6, cap_A=True):
        assert num_slices >= 3
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
        # TODO: Convert triangle_strip into a VBO object.
        #           Actually, do that in a separate method.
        if cap_A:
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
            return (triangle_strip, triangle_fan)
        return (triangle_strip, None)

    def draw_cylinder(tube, cap):

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, cap)

        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, 0, np.tile([1.,0.,0.], len(cap)))

        glDrawArrays(GL_TRIANGLE_FAN, 0, len(cap))

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, tube)

        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, 0, np.tile([0.,1.,1.], len(tube)))

        glDrawArrays(GL_TRIANGLE_STRIP, 0, len(tube))



class Sphere:
    @staticmethod
    def make_sphere(radius, slices, stacks):
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
        return (vertices, indices)

    @classmethod
    def draw(cls, coords, radius):
        vertices, indices = cls.make_sphere(radius, 10, 3)
        vertices += coords

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, vertices)

        # glEnableClientState(GL_COLOR_ARRAY)
        # glColorPointer(3, GL_FLOAT, 0, np.tile([1.,0.,0.], len(vertices)))

        print(len(vertices))
        print(max(indices.flat))
        assert max(indices.flat) < len(vertices)

        glDrawElements(GL_TRIANGLES, 3 * len(indices), GL_UNSIGNED_INT, indices);


def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    glEnable(GL_DEPTH_TEST)
    glShadeModel(GL_FLAT) # Or "GL_SMOOTH"
    glDisable(GL_CULL_FACE)
    # glEnable(GL_CULL_FACE)

    glTranslatef(0.0,0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(1, 3, 1, 1)
        glClearColor(0,0,0,0) # Background color.
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # Cylinder.draw_cylinder(
        #         *Cylinder.calculate_vertexes(
        #                 np.array([0.,0.,0.]),
        #                 np.array([0.,0,2]),
        #                 .3,
        #                 cap_A=True))

        Sphere.draw(np.array([0., 1.0, 0.]), .5)



        pygame.display.flip()
        pygame.time.wait(10)


main()
