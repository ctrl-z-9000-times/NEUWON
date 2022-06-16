from .rotate_align import rotate_align
import numpy as np

class Primative:
    __slots__ = ('vertices', 'indices')
    def get_vertices(self):
        assert self.vertices.dtype == np.float32
        return self.vertices
    def get_indices(self):
        assert self.indices.dtype == np.uint32
        return self.indices

class Sphere(Primative):
    __slots__ = ()
    def __init__(self, center, radius, slices):
        stacks = slices
        # Calculate the Vertices
        num_vertices = (stacks + 1) * (slices + 1)
        self.vertices = vertices = np.empty((num_vertices, 3), dtype=np.float32)
        write_idx = 0
        for i in range(stacks + 1):
            phi = np.pi * (i / stacks)
            # Loop Through Slices
            for j in range(slices + 1):
                theta = 2 * np.pi * (j / slices)
                # Calculate The Vertex Positions
                vertices[write_idx, 0] = np.cos(theta) * np.sin(phi)
                vertices[write_idx, 1] = np.cos(phi)
                vertices[write_idx, 2] = np.sin(theta) * np.sin(phi)
                write_idx += 1
        vertices *= radius
        vertices += center
        # Calculate The Index Positions
        num_triangles = 2 * slices * (stacks + 1)
        self.indices  = indices = np.zeros((num_triangles, 3), dtype=np.uint32)
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

class Cylinder(Primative):
    __slots__ = ()
    def __init__(self, A, B, diameter, num_slices):
        vertices = np.empty((2 * (num_slices + 1), 3), dtype=np.float32)
        vector = B - A
        length = np.linalg.norm(vector)
        vector /= length
        ref_vector = np.array([0.0, 0.0, 1.0])
        rot_matrix = rotate_align(vector, ref_vector)
        for s in range(num_slices + 1):
            f = 2 * np.pi * (s / num_slices)
            y = np.sin(f)
            x = np.cos(f)
            vertices[2*s,   0] = x
            vertices[2*s,   1] = y
            vertices[2*s,   2] = 0.
            vertices[2*s+1, 0] = x
            vertices[2*s+1, 1] = y
            vertices[2*s+1, 2] = length
        vertices = vertices.dot(rot_matrix)
        vertices += A
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = indices = np.empty((2 * num_slices, 3), dtype=np.uint32)
        for i in range(num_slices):
            indices[2*i,   2] = i * 2
            indices[2*i,   1] = i * 2 + 1
            indices[2*i,   0] = i * 2 + 2
            indices[2*i+1, 2] = i * 2 + 3
            indices[2*i+1, 1] = i * 2 + 2
            indices[2*i+1, 0] = i * 2 + 1

class Disk(Primative):
    __slots__ = ()
    def __init__(self,):
        1/0 # todo maybe?
        # Do not use triangle fans! Use regular triangles so that this conforms
        # to the same API as the other primatives.
        triangle_fan = np.empty((num_slices, 3), dtype=np.float32)
        for s in range(num_slices):
            f = 2 * np.pi * (s / num_slices)
            y = np.sin(f)
            x = np.cos(f)
            triangle_fan[s, 0] = x
            triangle_fan[s, 1] = y
            triangle_fan[s, 2] = 0.
        triangle_fan = triangle_fan.dot(rot_matrix)
        triangle_fan += A
