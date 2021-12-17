import numpy as np
import scipy.spatial
import time
import random
from neuwon.database import Real, epsilon, Pointer, NULL

from neuwon.outside.voronoi import *

um  = 1e-6    # Length in micrometers.
um2 = um ** 2 # Surface Area in square micrometers.
um3 = um ** 3 # Volume in cubic micrometers.

origin = np.zeros(3, dtype=Real)

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
    locations = np.array(locations, dtype=Pointer)
    compare_implementations(home, maximum_extent, locations, coordinates.reshape(-1,3))

def test_cubes_1():
    cubes_in_cubes(10, 10*um, 0., 10*um)
def test_cubes_2():
    cubes_in_cubes(50, 5*um, 0.000000001 *um, 23*um)
def test_cubes_3():
    cubes_in_cubes(200, 2*um, 1*um, 100*um)

def random_locations(arena_size, num_points, maximum_extent):
    print("Random Locations (%d)"%num_points)
    coordinates = np.random.uniform(0, arena_size, size=(num_points, 3))
    coordinates = np.array(coordinates, dtype=Real)
    locations = np.arange(num_points - 1, dtype=Pointer)
    compare_implementations(num_points - 1, maximum_extent, locations, coordinates)

def test_random_1():
    random_locations(100*um, 100, 50*um)
def test_random_2():
    random_locations(100*um, 10*1000, 50*um)
def test_random_3():
    random_locations(100*um, 1000*1000, 50*um)
