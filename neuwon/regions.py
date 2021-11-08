""" Tools for specifying 3-Dimensional volumes

This file provides tools for performing constructive solid geometry:
  * Geometric primitives: Rectangle, Sphere, Cylinder.
  * Logical operators for combining regions: Intersection, Union, Not.
  * The abstract class "Region" allows for defining new types of 3-D volumes.
"""
from collections.abc import Callable, Iterable, Mapping
import numpy as np

class Region:
    """ Abstract class for representing the shapes of 3-Dimensional volumes.

    Region subclasses must implement the following abstract methods:
      * Region.contains(self, coordinates) -> bool
      * Region.aabb(self) -> (lower_corner, upper_corner)
    """
    def contains(self, coordinates):
        """ Returns bool: does this region contain the given coordinates? """
    def aabb(self):
        """ Returns pair (lower_corner, upper_corner) of an axis aligned
        bounding box which entirely contains this region. """
    def sample_point(self):
        """ Returns a random point from within the region. """
        lower, upper = self.aabb()
        if not (all(np.isfinite(lower)) and all(np.isfinite(upper))):
            raise TypeError("Region is infinite!")
        while True:
            x = np.add(lower, np.random.uniform(size=(3)) * np.subtract(upper, lower))
            if self.contains(x):
                return x
    def sample_points(self, density):
        """ Returns a list of points from within the region. Density is
        approximate and is specified as points per unit of length cubed. """
        lower, upper = self.aabb()
        if not (all(np.isfinite(lower)) and all(np.isfinite(upper))):
            raise TypeError("Region is infinite!")
        aabb_volume = np.product(np.subtract(upper, lower))
        num_points = int(round(density * aabb_volume))
        points = np.add(lower, np.random.uniform(size=(num_points, 3)) * np.subtract(upper, lower))
        return points[[self.contains(x) for x in points]]

class Intersection(Region):
    """ Intersection of regions """
    def __init__(self, *regions):
        self.regions = regions
        assert(all(isinstance(rgn, Region) for rgn in self.regions))
        low, high = zip(*(rgn.aabb() for rgn in self.regions))
        self.lower_corner = np.max(low, axis=0)
        self.upper_corner = np.min(high, axis=0)
    def aabb(self):
        return (self.lower_corner, self.upper_corner)
    def contains(self, coordinates):
        return all(rgn.contains(coordinates) for rgn in self.regions)

class Union(Region):
    """ Union of regions """
    def __init__(self, *regions):
        self.regions = regions
        assert(all(isinstance(rgn, Region) for rgn in self.regions))
        low, high = zip(*(rgn.aabb() for rgn in self.regions))
        self.lower_corner = np.min(low, axis=0)
        self.upper_corner = np.max(high, axis=0)
    def aabb(self):
        return (self.lower_corner, self.upper_corner)
    def contains(self, coordinates):
        return any(rgn.contains(coordinates) for rgn in self.regions)

class Not(Region):
    """ Region covering everywhere *except* for the given region. """
    def __init__(self, region):
        """ Argument region is an instance of a subclass of Region. """
        self.region = region
        assert(isinstance(self.region, Region))
    def aabb(self):
        return ([-np.inf]*3, [+np.inf]*3)
    def contains(self, coordinates):
        return not self.region.contains(coordinates)

class Rectangle(Region):
    """ Axis Aligned Rectangular Prism """
    def __init__(self, corner1, corner2):
        """ Arguments corner1 and corner2 are the coordinates of any
        opposing corners of the box. """
        self.a = np.minimum(corner1, corner2)
        self.b = np.maximum(corner1, corner2)
        assert(len(self.a) == 3 and len(self.b) == 3)
    def aabb(self):
        return (self.a, self.b)
    def contains(self, coordinates):
        return all(self.a <= coordinates) and all(self.b > coordinates)

class Sphere(Region):
    def __init__(self, center, radius):
        """ Argument center is the 3D coordinates of the center of the sphere.
        Argument radius is distance from the center to include inside the sphere. """
        self.center = np.array(center, dtype=float)
        self.radius = float(radius)
        assert(len(self.center) == 3)
        assert(self.radius >= 0)
    def aabb(self):
        return (self.center - self.radius, self.center + self.radius)
    def contains(self, coordinates):
        return np.linalg.norm(self.center - coordinates) <= self.radius

class _Cone(Region):
    """ Right circular cone """
    def __init__(self, point, cap, radius):
        """ """
        self.point  = np.array(point, dtype=float)
        self.cap    = np.array(cap, dtype=float)
        self.radius = float(radius)
        assert(len(self.point) == 3)
        assert(len(self.cap) == 3)
        assert(self.radius >= 0)
    def aabb(self):
        return 1/0 # TODO
    def contains(self, coordinates):
        return 1/0 # TODO

class Cylinder(Region):
    """ Right circular cylinder """
    def __init__(self, point1, point2, radius):
        self.point1 = np.array(point1, dtype=float)
        self.point2 = np.array(point2, dtype=float)
        self.radius = float(radius)
        assert(len(self.point1) == 3)
        assert(len(self.point2) == 3)
        assert(self.radius >= 0)
        self.lower = np.minimum(self.point1, self.point2)
        self.upper = np.maximum(self.point1, self.point2)
        sqr = (self.point1 - self.point2) ** 2
        for dim in range(3):
            k = np.sqrt(sum(x for d, x in enumerate(sqr) if d != dim) / sum(sqr))
            self.lower[dim] -= k * self.radius
            self.upper[dim] += k * self.radius
        self.axis = self.point2 - self.point1
        self.length_sqr = sum(self.axis ** 2)
        self.radius_sqr = self.radius ** 2
    def aabb(self):
        return (self.lower, self.upper)
    def contains(self, coordinates):
        # https://flipcode.com/archives/Fast_Point-In-Cylinder_Test.shtml
        displacement = np.subtract(coordinates, self.point1)
        dot = self.axis.dot(displacement)
        if dot < 0 or dot > self.length_sqr:
            return False
        dist_sqr = displacement.dot(displacement) - dot*dot/self.length_sqr;
        return dist_sqr <= self.radius_sqr

class _Image(Rectangle):
    """ """
    def __init__(self, corner1, corner2, image, z_axis):
        """ """
        super().__init__(corner1, corner2)
        if   z_axis in "xX": self.z_axis = 0
        elif z_axis in "yY": self.z_axis = 1
        elif z_axis in "zZ": self.z_axis = 2
        else: self.z_axis = int(z_axis)
        # 
        self.image = np.array(image, dtype=np.float32)

    def contains(self, coordinates):
        if not super().contains(coordinates):
            return False
        # Use hashes to assign a stable unique identity to every location.
        h = hash(tuple(coordinates))
        # Convert hash into float in range [0, 1]
        z = 2 ** 30
        h = (h % z) / z
        # Get the image pixel value for these coordinates.
        shape   = self.image.shape
        x, y    = (dim for dim in range(3) if dim != self.z_axis)
        offset  = coordinates - self.a
        offset  = (offset[x], offset[y])
        scale   = ((self.b[x] - self.a[x]) / shape[0],
                   (self.b[y] - self.a[y]) / shape[1])
        coords  = (offset[0] / scale[0], offset[1] / scale[1])
        coords  = (min(int(coords[0]), shape[0]), min(int(coords[1]), shape[1]))
        pixel   = self.image[coords[0], coords[1]]
        # Decide if the location is part of this region.
        return h < pixel

class _ImageStack:
    """ """
    def __init__(self, stack):
        """ """
        1/0
        self.stack = Union([
            Image()
            for x in stack
            ])

    def aabb(self):
        return self.stack.aabb()

    def contains(self, coordinates):
        return self.stack.contains(coordinates)

class _RegionFactory(dict):
    def __init__(self, parameters: dict):
        super().__init__()
        self.add_parameters(parameters)

    def add_parameters(self, parameters: dict):
        self.parameters = parameters
        for name, rgn in self.parameters.items():
            self.add_region(name, rgn)
        del self.parameters

    def add_region(self, name: str, region_parameters) -> Region:
        if name in self:
            return self[name]
        self[name] = rgn = self.make_region(region_parameters)
        return rgn

    _region_types = {cls.__name__: cls for cls in
            (Intersection, Union, Not,
            Rectangle, Sphere, Cylinder)}

    def make_region(self, args) -> Region:
        if isinstance(args, Region):
            return args
        elif isinstance(args, str):
            region_name = args
            if region_name not in self:
                region_parameters = self.parameters[region_name]
                self.add_region(region_name, region_parameters)
            return self[region_name]
        elif isinstance(args, Iterable):
            region_type, *args = args
            if region_type in ('Intersection', 'Union', 'Not'):
                args = [self.make_region(r) for r in args]
            return self._region_types[region_type](*args)
        else:
            raise ValueError(args)
