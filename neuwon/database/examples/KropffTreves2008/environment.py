import math
import matplotlib.pyplot as plt
import random

class Environment:
    """ The environment is a 2D circle, in first quadrant with corner at origin. """
    def __init__(self, size):
        self.size     = size
        self.position = (size/2, size/2)
        self.speed    = 2.0 ** .5
        self.angle    = 0
        self.course   = []

    def in_bounds(self, position):
        x, y = position
        if True:
            # Circle arena
            radius = self.size / 2.
            hypot  = (x - radius) ** 2 + (y - radius) ** 2
            return hypot < radius ** 2
        else:
            # Square arena
            x_in = x >= 0 and x < self.size
            y_in = y >= 0 and y < self.size
            return x_in and y_in

    def move(self):
        max_rotation = 2 * math.pi / 20
        self.angle += random.uniform(-max_rotation, max_rotation)
        self.angle = (self.angle + 2 * math.pi) % (2 * math.pi)
        vx = self.speed * math.cos(self.angle)
        vy = self.speed * math.sin(self.angle)
        x, y = self.position
        new_position = (x + vx, y + vy)
        if self.in_bounds(new_position):
            self.position = new_position
            self.course.append(self.position)
        else:
            # On failure, recurse and try again.
            assert(self.in_bounds(self.position))
            self.angle = random.uniform(0, 2 * math.pi)
            self.move()

    def plot_course(self, show=True):
        plt.figure("Path")
        plt.ylim([0, self.size])
        plt.xlim([0, self.size])
        self.course = self.course[:100*1000]
        x, y = zip(*self.course)
        plt.plot(x, y, 'k-')
        if show: plt.show()
