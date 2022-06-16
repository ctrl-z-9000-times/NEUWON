from .primatives import Sphere, Cylinder
from neuwon.database import Pointer, NULL
from OpenGL.GL import *
from OpenGL.GLU import *
from scipy.spatial.transform import Rotation
import numpy as np

class Camera:
    def __init__(self, size, fov, dist):
        """
        Argument size is the window size.
        Argument fov is the angle of the field of view.
        Argument dist is the maximum view distance.
        """
        self.size    = (int(size[0]), int(size[1]))
        self.fov     = float(fov)
        self.dist    = float(dist)
        self.pos     = np.zeros(3)
        self.pitch   = 0.0
        self.yaw     = 0.0
        self.up      = np.array([ 0.0, 1.0, 0.0]) # +Y is up.
        self.forward = np.array([ 0.0, 0.0, -1.0])
        self.update_rotation_matrix()

    def update_rotation_matrix(self):
        self.rotation = (
                Rotation.from_euler('x', (self.pitch)) *
                Rotation.from_euler('y', (self.yaw))
        ).as_matrix()

    def move(self, offset):
        """
        Move the camera position.
        The offset is relative to the camera's viewpoint, not the world.
        """
        self.pos += np.array(offset).dot(self.rotation)

    def setup_opengl(self):
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity();
        gluPerspective(self.fov, (self.size[0]/self.size[1]), 0.1, self.dist)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        lookat = self.forward.dot(self.rotation) + self.pos
        gluLookAt(*self.pos, *lookat, *self.up)

    def print(self):
        print("Camera Position", self.pos)
        print("Camera Pitch",    self.pitch)
        print("Camera Yaw  ",    self.yaw)

class Scene:
    """ This class generates the 3D mesh and renders it using OpenGL. """
    def __init__(self, parent, coordinates, diameter, lod=2.5):
        self._init_opengl_settings()
        # Collect all of the 3D objects, one per segment.
        self.num_seg = num_seg = len(parent)
        self.objects = objects = np.zeros(num_seg, dtype=object)
        for idx in range(num_seg):
            p       = parent[idx]
            coords  = coordinates[idx]
            diam    = diameter[idx]
            nslices = max(3, int(lod * diam))
            if p == NULL:
                objects[idx] = Sphere(coords, 0.5 * diam, nslices)
            else:
                p_coords = coordinates[p]
                objects[idx] = Cylinder(coords, p_coords, diam, nslices)
        # Render all segments until told to do otherwise.
        self.compile_visible(np.arange(num_seg, dtype=Pointer))

    def _init_opengl_settings(self):
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_FLAT) # Or "GL_SMOOTH"
        glDisable(GL_CULL_FACE)
        # Setup transparency for overlaying text.
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def compile_visible(self, visible_segments):
        """ Combine all of the visible objects into one big object. """
        segment_idx = []
        vertices    = []
        indices     = []
        for idx in visible_segments:
            obj = self.objects[idx]
            segment_idx.append(idx)
            vertices.append(obj.get_vertices())
            indices.append(np.copy(obj.get_indices()))
        if not segment_idx:
            self.vertices = np.empty(0, dtype=np.float32)
            self.indices  = np.empty(0, dtype=np.uint32)
            self.segments = np.empty(0, dtype=Pointer)
            return
        # Offset the indices to point to the new vertex locations in the combined buffer.
        num_v = np.cumsum([len(x) for x in vertices])
        for v_index, v_offset in zip(indices[1:], num_v):
            v_index += v_offset
        self.vertices = np.vstack(vertices)
        self.indices  = np.vstack(indices)
        # Record which segment generated each vertex.
        self.segments = np.empty(len(self.vertices), dtype=Pointer)
        lower = 0
        for obj_idx, seg_idx in enumerate(segment_idx):
            upper = num_v[obj_idx]
            self.segments[lower:upper] = seg_idx
            lower = upper

        # TODO: Move these arrays to the GPU now, instead of copying them at
        # render time. Use VBO's?

    def draw(self, camera, segment_colors, background_color):
        camera.setup_opengl()

        # Draw background.
        glClearColor(*background_color, 0.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)

        assert len(segment_colors) == self.num_seg, "Model changed, but 3d mesh did not!"
        colors = np.take(segment_colors, self.segments, axis=0)
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_FLOAT, 0, colors)

        glDrawElements(GL_TRIANGLES, 3 * len(self.indices), GL_UNSIGNED_INT, self.indices)

    def get_segment(self, camera, screen_coordinates):

        camera.setup_opengl()
        x, y = screen_coordinates
        y = camera.size[1] - y

        background = 2**24 - 1
        assert self.num_seg < background

        colors = np.empty((len(self.vertices), 3), dtype=np.uint8)
        mask = 2**8 - 1
        np.bitwise_and(self.segments, mask,       out=colors[:,0])
        np.bitwise_and(self.segments, mask << 8,  out=colors[:,1])
        np.bitwise_and(self.segments, mask << 16, out=colors[:,2])

        # Only render a single pixel.
        glScissor(x, y, 1, 1)
        glEnable(GL_SCISSOR_TEST)

        glClearColor(1.0, 1.0, 1.0, 0.0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        glEnableClientState(GL_VERTEX_ARRAY)
        glVertexPointer(3, GL_FLOAT, 0, self.vertices)
        glEnableClientState(GL_COLOR_ARRAY)
        glColorPointer(3, GL_UNSIGNED_BYTE, 0, colors)
        glDrawElements(GL_TRIANGLES, 3 * len(self.indices), GL_UNSIGNED_INT, self.indices)

        glFlush()
        glFinish()
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        color = glReadPixels(x, y, 1, 1, GL_RGB, GL_UNSIGNED_BYTE)
        segment_index = color[0] + (color[1] << 8) + (color[2] << 16)

        # Restore OpenGL settings.
        glDisable(GL_SCISSOR_TEST)

        if segment_index == background:
            return NULL
        else:
            return segment_index
