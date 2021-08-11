import math
import numpy as np
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

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

def Cylinder(A, B, diameter, num_slices=5, cap_A=False):
    assert num_slices >= 3
    triangle_strip = np.empty((2 * (num_slices + 1), 3), dtype=np.float32)
    vector = B - A
    length = np.linalg.norm(vector)
    vector /= length
    ref_vector = np.array([0, 0, 1], dtype=np.float32)
    for s in range(num_slices + 1):
        f = 2 * math.pi * (s / num_slices)
        y = math.sin(f)
        x = math.cos(f)
        triangle_strip[2*s,   0] = x
        triangle_strip[2*s,   1] = y
        triangle_strip[2*s,   2] = 0
        triangle_strip[2*s+1, 0] = x
        triangle_strip[2*s+1, 1] = y
        triangle_strip[2*s+1, 2] = length
    triangle_strip = triangle_strip.dot(_rotateAlign(ref_vector, vector))
    triangle_strip += A
    # TODO: Convert triangle_strip into a VBO object.
    #           Actually, do that in a separate method.
    if cap_A:
        triangle_fan = np.empty((num_slices, 3), dtype=np.float32)
        for s in range(num_slices):
            f = 2 * math.pi * (s / num_slices)
            y = math.sin(f)
            x = math.cos(f)
            triangle_fan[s, 0] = 
            triangle_fan[s, 1] = 
            triangle_fan[s, 2] = 
            return (triangle_strip, triangle_fan)
    return triangle_strip

def draw_cylinder(triangle_strip):
    glEnableClientState(GL_VERTEX_ARRAY)
    glVertexPointer(3, GL_FLOAT,0,triangle_strip)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, len(triangle_strip))
    glDisableClientState(GL_VERTEX_ARRAY)

    # glEnableClientState(GL_COLOR_ARRAY)
    # glColorPointer(3, GL_FLOAT,0,c)


def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)

    glTranslatef(0.0,0.0, -5)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        glRotatef(1, 3, 1, 1)
        glClearColor(0,0,0,0) # Background color.
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        draw_cylinder(Cylinder(np.array([0.,0.,0.]), np.array([0.,0,1]), .3))

        pygame.display.flip()
        pygame.time.wait(10)


main()
