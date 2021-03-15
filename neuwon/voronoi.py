import numpy as np
import numba
import numba.types
import math
import scipy.spatial
import time
import random
from neuwon.common import Real, epsilon, Location, ROOT

__all__ = ["voronoi_cell"]

origin = np.zeros(3, dtype=Real)

um  = 1e-6    # Length in micrometers.
um2 = um ** 2 # Surface Area in square micrometers.
um3 = um ** 3 # Volume in cubic micrometers.

Plane = np.dtype([
    ("normal", Real, (3,)),
    ("offset", Real),
])

Triangle = np.dtype([
    ("vertexes", Real, (3, 3)),
    ("facing_location", Location),
    ("max_dist_sqr", Real),
    ("surface_area", Real),
])

Neighbor = np.dtype([
    ("location", Location),
    ("distance", Real),
    ("border_surface_area", Real),
])

Real_t      = numba.from_dtype(Real)
Location_t  = numba.from_dtype(Location)
Vec_t       = Real_t[::1] # Contiguous.
Void_t      = numba.void
Boolean_t   = numba.boolean
Tuple_t     = numba.types.Tuple
Plane_t     = numba.from_dtype(Plane)
Triangle_t  = numba.from_dtype(Triangle)
Neighbor_t  = numba.from_dtype(Neighbor)

@numba.njit(Real_t(Real_t[:]), cache=True)
def magnitude(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)

@numba.njit(Void_t(Plane_t, Vec_t, Vec_t), cache=True)
def new_plane(plane, under, over):
    for i in range(3):
        plane["normal"][i] = over[i] - under[i]
    distance = magnitude(plane["normal"])
    plane["offset"] = 0.5 * distance
    for i in range(3):
        plane["normal"][i] *= (1.0 / distance)

@numba.njit(Boolean_t(Plane_t, Vec_t), cache=True)
def plane_contains_point(plane, point):
    """ Is the point contained within or below the plane? """
    return plane["normal"].dot(point) <= plane["offset"] + epsilon * 1e-6

@numba.njit(Vec_t(Plane_t, Vec_t, Vec_t), cache=True)
def plane_line_intersection(plane, a, b):
    intersection = b - a
    denom = plane["normal"].dot(intersection)
    numer = plane["offset"] - plane["normal"].dot(a)
    x = max(0.0, min(1.0, (numer / denom)))
    for i in range(3):
        intersection[i] = x * intersection[i] + a[i]
    return intersection

@numba.njit(Void_t(Triangle_t, Real_t[:], Real_t[:], Real_t[:], Location_t), cache=True)
def new_triangle(triangle, v0, v1, v2, facing_location):
    triangle["vertexes"][0] = v0
    triangle["vertexes"][1] = v1
    triangle["vertexes"][2] = v2
    triangle["facing_location"] = facing_location
    vector_ab = triangle["vertexes"][1] - triangle["vertexes"][0]
    vector_ac = triangle["vertexes"][2] - triangle["vertexes"][0]
    triangle["surface_area"] = 0.5 * magnitude(np.cross(vector_ab, vector_ac))
    triangle["max_dist_sqr"] = 0
    for v in range(3):
        dist = 0
        for d in range(3):
            dist += triangle["vertexes"][v][d] ** 2
        triangle["max_dist_sqr"] = max(triangle["max_dist_sqr"], dist)

@numba.njit(Real_t(Triangle_t), cache=True)
def triangle_volume(triangle):
    """ Compute the volume of the triangular pyramid formed by this
    triangle and the origin. """
    a, b, c = triangle["vertexes"]
    vector_b = b - a
    vector_c = c - a
    vector_home = origin - a
    base_cross = np.cross(vector_b, vector_c)
    base_cross_magnitude = magnitude(base_cross)
    if base_cross_magnitude == 0.0: # TODO: Consider if there are any floating point issues here?
        return 0.0
    base_cross *= 1.0 / base_cross_magnitude
    height = base_cross.dot(vector_home)
    return abs(height * base_cross_magnitude / 6.0)

@numba.njit(Neighbor_t[::1](Triangle_t[::1]), cache=True)
def triangles_to_neighbors(triangles):
    neighbors_array = np.empty(len(triangles), dtype=Neighbor)
    for i in range(len(triangles)):
        neighbors_array[i]["location"] = triangles[i]["facing_location"]
        neighbors_array[i]["border_surface_area"] = triangles[i]["surface_area"]
    # Consolidate all of the neighbor entries for each facing_location.
    # Secondary sort key is border_surface_area to avoid sorting by uninitialized data.
    with numba.objmode():
        neighbors_array.sort(order=["location", "border_surface_area"])
    previous = ROOT # Detect boundaries of contiguous blocks of each facing_location.
    write_idx = -1 # Scan and compress in place.
    for neighbor in neighbors_array:
        if neighbor["location"] == ROOT:
            break
        elif neighbor["location"] != previous:
            write_idx += 1
            neighbors_array[write_idx]["location"] = neighbor["location"]
            neighbors_array[write_idx]["border_surface_area"] = neighbor["border_surface_area"]
            previous = neighbor["location"]
        else:
            neighbors_array[write_idx]["border_surface_area"] += neighbor["border_surface_area"]
    return neighbors_array[:write_idx+1]

@numba.njit(Triangle_t[::1](Triangle_t[::1], Plane_t, Location_t), cache=True)
def add_plane(triangles, plane, facing_location):
    len_new_triangles = 0
    new_triangles = np.empty(3 * len(triangles) - 1, dtype=Triangle)
    len_new_vertex_pairs = 0
    new_vertex_pairs = np.empty((len(triangles), 2, 3), dtype=Real)
    """ Removes all areas of the convex hull which are above the given plane. """
    for t in triangles:
        # Check which vertexes are still contained in the convex hull.
        num_alive = 0
        alive1 = 0; alive2 = 0
        dead1 = 0; dead2 = 0
        for v in range(3):
            if plane_contains_point(plane, t["vertexes"][v]):
                if num_alive == 0:
                    alive1 = v
                else:
                    alive2 = v
                num_alive += 1
            else:
                if num_alive == v:
                    dead1 = v
                else:
                    dead2 = v
        # Cut the triangles where it crosses above the plane. Put the
        # results into new_triangles.
        if num_alive == 3:
            # Triangle is untouched by the plane. Return it unchanged.
            new_triangles[len_new_triangles] = t; len_new_triangles += 1
        elif num_alive == 0:
            # Triangle is entirely above the plane. Do not return it.
            pass
        elif num_alive == 1:
            # Two corners removed. Modify this triangle by cutting off one edge.
            new1 = plane_line_intersection(plane, t["vertexes"][alive1], t["vertexes"][dead1])
            new2 = plane_line_intersection(plane, t["vertexes"][alive1], t["vertexes"][dead2])
            new_vertex_pairs[len_new_vertex_pairs][0] = new1
            new_vertex_pairs[len_new_vertex_pairs][1] = new2
            len_new_vertex_pairs += 1
            new_triangle(new_triangles[len_new_triangles],
                new1, new2, t["vertexes"][alive1], t["facing_location"])
            if new_triangles[len_new_triangles]["surface_area"] > epsilon * um2:
                len_new_triangles += 1
        elif num_alive == 2:
            # One corner removed. Cut off the corner turning this into a
            # quadrilateral. Break into two new triangles.
            new1 = plane_line_intersection(plane, t["vertexes"][alive1], t["vertexes"][dead1])
            new2 = plane_line_intersection(plane, t["vertexes"][alive2], t["vertexes"][dead1])
            new_vertex_pairs[len_new_vertex_pairs][0] = new1
            new_vertex_pairs[len_new_vertex_pairs][1] = new2
            len_new_vertex_pairs += 1
            #
            new_triangle(new_triangles[len_new_triangles],
                t["vertexes"][alive1], t["vertexes"][alive2], new1, t["facing_location"])
            if new_triangles[len_new_triangles]["surface_area"] > epsilon * um2:
                len_new_triangles += 1
            #
            new_triangle(new_triangles[len_new_triangles],
                new1, new2, t["vertexes"][alive2], t["facing_location"])
            if new_triangles[len_new_triangles]["surface_area"] > epsilon * um2:
                len_new_triangles += 1
    # Triangulate the surface of the new plane like a fan.
    if len_new_vertex_pairs > 0:
        anchor = new_vertex_pairs[0][0]
        for b, c in new_vertex_pairs[1:len_new_vertex_pairs]:
            new_triangle(new_triangles[len_new_triangles],
                anchor, b, c, facing_location)
            if new_triangles[len_new_triangles]["surface_area"] > epsilon * um2:
                len_new_triangles += 1
    return new_triangles[:len_new_triangles]

@numba.njit(Triangle_t[::1](Real_t), cache=True)
def sphere(r):
    """ Where pi == 4 """
    corners = np.empty((8, 3), dtype=Real)
    for x in range(8):
        for d in range(3):
            if x & (1 << d) == 0:
                corners[x, d] = -r
            else:
                corners[x, d] = r
    triangles = np.empty(12, dtype=Triangle)
    # Dimension 0 -
    new_triangle(triangles[0], corners[0], corners[6], corners[2], ROOT)
    new_triangle(triangles[1], corners[0], corners[6], corners[4], ROOT)
    # Dimension 0 +
    new_triangle(triangles[2], corners[1], corners[7], corners[3], ROOT)
    new_triangle(triangles[3], corners[1], corners[7], corners[5], ROOT)
    # Dimension 1 -
    new_triangle(triangles[4], corners[0], corners[1], corners[4], ROOT)
    new_triangle(triangles[5], corners[1], corners[4], corners[5], ROOT)
    # Dimension 1 +
    new_triangle(triangles[6], corners[2], corners[3], corners[6], ROOT)
    new_triangle(triangles[7], corners[3], corners[6], corners[7], ROOT)
    # Dimension 2 -
    new_triangle(triangles[8], corners[0], corners[2], corners[3], ROOT)
    new_triangle(triangles[9], corners[3], corners[1], corners[0], ROOT)
    # Dimension 2 +
    new_triangle(triangles[10], corners[7], corners[4], corners[6], ROOT)
    new_triangle(triangles[11], corners[4], corners[7], corners[5], ROOT)
    return triangles

@numba.njit(Tuple_t((Real_t, Neighbor_t[::1]))(Location_t, Real_t, Location_t[:], Real_t[:, ::1]), cache=True)
def voronoi_cell(home_location, maximum_extent, neighbor_locations, coordinates):
    """ Returns pair of (volume, neighbors) where neighbors is an array with
    data type Neighbor. """
    triangles = sphere(maximum_extent)
    planes = np.empty(len(neighbor_locations), dtype=Plane)
    for i, n in enumerate(neighbor_locations):
        new_plane(planes[i], coordinates[home_location], coordinates[n])
    with numba.objmode(order='intp[:]'):
        order = np.argsort(planes, order="offset")
    for i in order:
        if np.max(triangles["max_dist_sqr"]) < planes[i]["offset"] ** 2:
            break
        triangles = add_plane(triangles, planes[i], neighbor_locations[i])
    volume = 0.0
    for t in triangles:
        volume += triangle_volume(t)
    return (volume, triangles_to_neighbors(triangles))

def reference_implementation(home_coordinates, maximum_extent,
        neighbor_locations, neighbor_coordinates):
    """ This computes a voronoi cell using the qhull library, via scipy.
    Use this to double check the results of the primary implementation.
    Returns pair of (surface_area, volume). """
    bounding_sphere = np.array([ # pi == 4
        # Planes as [normal, offset]
        [ 1, 0,  0, -maximum_extent],
        [-1, 0,  0, -maximum_extent],
        [0,  1,  0, -maximum_extent],
        [0, -1,  0, -maximum_extent],
        [0,  0,  1, -maximum_extent],
        [0,  0, -1, -maximum_extent],
    ])
    midpoints = np.array((neighbor_coordinates - home_coordinates) / 2)
    midpoint_distances = np.linalg.norm(midpoints, axis=1)
    normals = midpoints / np.expand_dims(midpoint_distances, 1)
    offsets = np.sum(normals * midpoints, axis=1).reshape(-1,1)
    planes = np.vstack((np.hstack((normals, -offsets)), bounding_sphere))
    halfspace_hull = scipy.spatial.HalfspaceIntersection(planes, origin)
    convex_hull = scipy.spatial.ConvexHull(halfspace_hull.intersections)
    return (convex_hull.area, convex_hull.volume)

def compare_implementations(home_location, maximum_extent,
        neighbor_locations, coordinates):
    # Run the primary implementation.
    p_time = time.time()
    p_volume, p_neighbors = voronoi_cell(home_location, maximum_extent,
            neighbor_locations, coordinates)
    for n in p_neighbors:
        n["distance"] = np.linalg.norm(coordinates[home_location] - coordinates[n["location"]])
    p_time = time.time() - p_time
    p_sa = np.sum(p_neighbors["border_surface_area"])
    # Run the reference implementation.
    r_time = time.time()
    r_sa, r_volume = reference_implementation(coordinates[home_location], maximum_extent,
            neighbor_locations, coordinates[neighbor_locations])
    r_time = time.time() - r_time
    # Print results
    print("Primary implementation:   %g seconds"%p_time)
    print("Reference implementation: %g seconds"%r_time)
    print("Diff Volume %g um3"%(abs(p_volume - r_volume) / um3))
    print("Diff Surface Area %g um2"%(abs(p_sa - r_sa) / um2))
    print()

if __name__ == "__main__": # Run unit tests.
    def cubes_in_cubes(size, spacing, jitter, maximum_extent):
        print("Cubes (%d)"%(size**3))
        coordinates = np.empty([size,size,size,3], dtype=Real)
        for x in range(size): coordinates[x, :, :, 0] = x * spacing
        for y in range(size): coordinates[:, y, :, 1] = y * spacing
        for z in range(size): coordinates[:, :, z, 2] = z * spacing
        coordinates += np.random.uniform(-jitter, +jitter, size=coordinates.shape)
        locations = list(range(size ** 3))
        random.shuffle(locations)
        home = locations.pop()
        locations = np.array(locations, dtype=Location)
        compare_implementations(home, maximum_extent, locations, coordinates.reshape(-1,3))

    cubes_in_cubes(10, 10*um, 0., 10*um)
    cubes_in_cubes(50, 5*um, 0.000000001 *um, 23*um)
    cubes_in_cubes(200, 2*um, 1*um, 100*um)

    def random_locations(arena_size, num_points, maximum_extent):
        print("Random Locations (%d)"%num_points)
        coordinates = np.random.uniform(0, arena_size, size=(num_points, 3))
        coordinates = np.array(coordinates, dtype=Real)
        locations = np.arange(num_points - 1, dtype=Location)
        compare_implementations(num_points - 1, maximum_extent, locations, coordinates)

    random_locations(100*um, 100, 50*um)
    random_locations(100*um, 10*1000, 50*um)
    random_locations(100*um, 1000*1000, 50*um)
