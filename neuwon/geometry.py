import numpy as np
import scipy.spatial
from neuwon import Location, Real, epsilon

ROOT = np.iinfo(Location).max

class Neighbor:
    """ Adjacent location in the partitioning of the extracellular medium """
    __slots__ = ("location", "distance", "border_surface_area")

class Geometry:
    """ Physical shapes & structures of neurons """
    def __init__(self, coordinates, parents, diameters,
            maximum_extracellular_radius=100e-6,):
        # Save the arguments.
        self.coordinates = np.array(coordinates, dtype=Real)
        self.parents = np.array([ROOT if p is None else p for p in parents], dtype=Location)
        self.diameters = np.array(diameters, dtype=Real)
        self.maximum_extracellular_radius = float(maximum_extracellular_radius)
        # Check the arguments.
        assert(len(self.coordinates) == len(self))
        assert(len(self.parents)     == len(self))
        assert(len(self.diameters)   == len(self))
        assert(all(all(np.isfinite(c)) for c in self.coordinates))
        assert(all(p < len(self) or p == ROOT for p in self.parents))
        assert(all(d >= 0 for d in self.diameters))
        assert(self.maximum_extracellular_radius > epsilon * 1e-6)
        # Initialize the geometric properties.
        self._init_tree_properties()
        self._init_cellular_properties()
        self._init_extracellular_properties()

    def _init_tree_properties(self):
        # Compute the children lists.
        self.children = np.empty(len(self), dtype=object)
        for location in range(len(self)):
            self.children[location] = []
        for location, parent in enumerate(self.parents):
            if not self.is_root(location):
                self.children[parent].append(location)
        # Root must have at least one child, because cylinder is defined as between two points.
        assert(all(len(self.children[x]) >= 1 for x in range(len(self)) if self.is_root(x)))
        # The child with the largest diameter is special and is always kept at
        # the start of the children list.
        for siblings in self.children:
            siblings.sort(reverse=True, key=lambda x: self.diameters[x])
        # Compute lengths, which are the distances between each node and its
        # parent node. All root node lengths are NAN.
        self.lengths = np.empty(len(self), dtype=Real)
        for location in range(len(self)):
            if self.is_root(location):
                self.lengths[location] = np.nan
            else:
                self.lengths[location] = np.linalg.norm(
                    self.coordinates[location] - self.coordinates[self.parents[location]])
        assert(all(l >= epsilon * (1e-6)**1 or self.is_root(idx) for idx, l in enumerate(self.lengths)))

    def _init_cellular_properties(self):
        self.cross_sectional_areas = np.array([np.pi * (d / 2) ** 2 for d in self.diameters], dtype=Real)
        self.surface_areas = np.empty(len(self), dtype=Real)
        self.intra_volumes = np.empty(len(self), dtype=Real)
        for location, parent in enumerate(self.parents):
            radius = self.diameters[location] / 2
            if self.is_root(location):
                # Root of new tree. The body of this segment is half of the
                # cylinder spanning between this node and its first child.
                eldest = self.children[location][0]
                length = self.diameters[eldest] / 2
            elif self.is_root(parent) and self.children[parent][0] == location:
                length = self.lengths[location] / 2
            else:
                length = self.lengths[location]
            # Primary segments are straightforward extensions of the parent
            # branch. Non-primary segments are lateral branchs off to the side
            # of the parent branch. Subtract the parent's radius from the
            # secondary nodes length, to avoid excessive overlap between
            # segments.
            if self.is_root(location):
                primary = True
            else:
                siblings = self.children[parent]
                if siblings[0] == location or (self.is_root(parent) and siblings[1] == location):
                    primary = True
                else:
                    primary = False
            if not primary:
                parent_radius = self.diameters[parent] / 2
                if length > parent_radius + epsilon * 1e-6:
                    length -= parent_radius
                else:
                    # This segment is entirely enveloped within its parent. In
                    # this corner case allow the segment to protrude directly
                    # from the center of the parent instead of the surface.
                    pass
            self.surface_areas[location] = 2 * np.pi * radius * length
            self.intra_volumes[location] = np.pi * radius ** 2 * length
            # Account for the surface area on the tips of terminal/leaf segments.
            num_children = len(self.children[location])
            if num_children == 0 or (self.is_root(location) and num_children == 1):
                self.surface_areas[location] += np.pi * radius ** 2
        assert(all(x  >= epsilon * (1e-6)**2 for x in self.cross_sectional_areas))
        assert(all(sa >= epsilon * (1e-6)**2 for sa in self.surface_areas))
        assert(all(v  >= epsilon * (1e-6)**3 for v in self.intra_volumes))

    def _init_extracellular_properties(self):
        self._tree = scipy.spatial.cKDTree(self.coordinates)
        self.extra_volumes = np.empty(len(self), dtype=Real)
        self.neighbors = np.zeros(len(self), dtype=object)
        bounding_sphere = np.array([
                [ 1, 0,  0, -self.maximum_extracellular_radius],
                [-1, 0,  0, -self.maximum_extracellular_radius],
                [0,  1,  0, -self.maximum_extracellular_radius],
                [0, -1,  0, -self.maximum_extracellular_radius],
                [0,  0,  1, -self.maximum_extracellular_radius],
                [0,  0, -1, -self.maximum_extracellular_radius],
            ])
        origin = np.zeros(3)
        for location in range(len(self)):
            coords = self.coordinates[location]
            neighbors = self._tree.query_ball_point(coords, 2 * self.maximum_extracellular_radius)
            neighbors.remove(location)
            midpoints = np.array((self.coordinates[neighbors] - coords) / 2, dtype=float)
            normals = midpoints / np.linalg.norm(midpoints)
            offsets = np.sum(normals * midpoints, axis=1).reshape(-1,1)
            planes = np.vstack((np.hstack((normals, -offsets)), bounding_sphere))
            halfspace_hull = scipy.spatial.HalfspaceIntersection(planes, origin, qhull_options='t')
            convex_hull = scipy.spatial.ConvexHull(halfspace_hull.intersections)
            self.extra_volumes[location] = convex_hull.volume
            self.neighbors[location] = []
            for v in halfspace_hull.dual_vertices:
                if v not in range(len(neighbors)): continue # Adjacency to the bounding sphere.
                n = Neighbor()
                n.location = neighbors[v]
                n.distance = np.linalg.norm(coords - self.coordinates[n.location])
                # The scipy bindings to QHull don't appear expose the area of
                # each facet. Instead find the vertexes of this facet, project
                # the vertexes onto the facets plane, and recompute the convex
                # hull in 2D to get the surface area.
                basis1 = scipy.spatial.transform.Rotation.from_rotvec([np.pi/2, 0, 0]).apply(planes[v,:3])
                basis2 = scipy.spatial.transform.Rotation.from_rotvec([0, np.pi/2, 0]).apply(planes[v,:3])
                projection = []
                for x in halfspace_hull.intersections:
                    if abs(np.dot(x, planes[v,:3]) + planes[v,3]) <= epsilon:
                        projection.append([basis1.dot(x), basis2.dot(x)])
                facet_hull = scipy.spatial.ConvexHull(projection, qhull_options='t')
                n.border_surface_area = facet_hull.volume
                self.neighbors[location].append(n)

    def __len__(self):
        return len(self.coordinates)

    def is_root(self, location):
        return self.parents[location] == ROOT

    def nearest_neighbors(self, coordinates, k, maximum_distance=np.inf):
        coordinates = np.array(coordinates, dtype=Real)
        assert(coordinates.shape == (3,))
        assert(all(np.isfinite(x) for x in coordinates))
        k = int(k)
        assert(k >= 1)
        d, i = self._tree.query(coordinates, k, distance_upper_bound=maximum_distance)
        return i
