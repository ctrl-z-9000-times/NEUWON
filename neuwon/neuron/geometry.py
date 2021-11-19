from neuwon.database import epsilon, Compute, NULL
import numpy as np

@Compute
def surface_area_disk(diameter):
    """ Surface area of one side only, area of circle. """
    return 0.25 * np.pi * (diameter ** 2)

@Compute
def surface_area_sphere(diameter):
    return np.pi * (diameter ** 2)

@Compute
def surface_area_cylinder(diameter, length):
    """ Lateral surface area, does not include the end caps. """
    return np.pi * diameter * length

@Compute
def surface_area_frustum(radius_1, radius_2, length):
    """ Lateral surface area, does not include the end caps. """
    s = sqrt((radius_1 - radius_2) ** 2 + length ** 2)
    return np.pi * (radius_1 + radius_2) * s

@Compute
def volume_sphere(diameter):
    return (4.0 / 3.0 ) * np.pi * (0.5 * diameter) ** 3

@Compute
def volume_cylinder(diameter, length):
    return surface_area_disk(diameter) * length

@Compute
def volume_frustum(radius_1, radius_2, length):
    return np.pi / 3.0 * length * (radius_1 * radius_1 + radius_1 * radius_2 + radius_2 * radius_2)

class Tree:
    """
    Segments are organized in a tree.
    """
    __slots__ = ()
    @staticmethod
    def _initialize(database):
        db_cls = database.get_class('Segment')
        db_cls.add_attribute("parent", dtype=db_cls, allow_invalid=True)
        db_cls.add_connectivity_matrix("children", db_cls)

    def __init__(self, parent):
        self.parent = parent
        # Add ourselves to the parent's children list.
        parent = self.parent
        if parent is not None:
            siblings = parent.children
            siblings.append(self)
            parent.children = siblings

    @Compute
    def is_root(self) -> bool:
        return self.parent == NULL

class Geometry:
    """
    The root of the tree is a sphere,
    all other segments are cylinders.
    """
    __slots__ = ()
    @staticmethod
    def _initialize(database):
        db_cls = database.get_class('Segment')

        db_cls.add_attribute("coordinates", shape=(3,),
                units="μm",)

        db_cls.add_attribute("diameter",
                units="μm",
                valid_range=(0.0, np.inf),)

        db_cls.add_attribute("_primary", dtype=np.bool, doc="""
                Primary segments are straightforward extensions of the parent
                branch. Non-primary segments are lateral branches off to the side of
                the parent branch. """)

        db_cls.add_attribute("length",
                units="μm",
                valid_range=(epsilon, np.inf),
                doc="""The distance between this node and its parent node. """)

        db_cls.add_attribute("surface_area",
                units="μm²",
                valid_range=(epsilon, np.inf),)

        db_cls.add_attribute("cross_sectional_area",
                units="μm²",
                valid_range = (epsilon, np.inf),)

        db_cls.add_attribute("volume",
                units="μm³",
                valid_range=(epsilon, np.inf),)

    def __init__(self, coordinates, diameter):
        self.coordinates    = coordinates
        self.diameter       = diameter
        parent              = self.parent
        # Determine the _primary flag.
        if self.is_sphere():
            self._primary = False # Value does not matter.
        elif parent.is_sphere():
            self._primary = False # Spheres have no primary branches off of them.
        else:
            # Set the first child added to a segment as the primary extension,
            # and all subsequent children as secondary branches.
            self._primary = len(parent.children) < 2

        self._compute_length()
        self._compute_surface_area()
        if parent: parent._compute_surface_area()
        self._compute_cross_sectional_area()
        self._compute_intracellular_volume()

    @Compute
    def is_sphere(self) -> bool:
        return self.is_root()

    @Compute
    def is_cylinder(self) -> bool:
        return not self.is_sphere()

    @Compute
    def _compute_length(self):
        if self.is_sphere():
            # Spheres have no defined length, so make one up instead instead.
            self.length = (2.0 / 3.0) * self.diameter
        else:
            self.length = np.linalg.norm(self.coordinates - self.parent.coordinates)

    @Compute
    def _secondary_length(self) -> float:
        """
        Subtract the parent's radius from the secondary nodes length,
        to avoid excessive overlap between segments.
        """
        length = self.length
        if not self._primary:
            parent_radius = 0.5 * self.parent.diameter
            if length < parent_radius + epsilon:
                # This segment is entirely enveloped within its parent. In
                # this corner case allow the segment to protrude directly
                # from the center of the parent instead of the surface.
                pass
            else:
                length -= parent_radius
        return length

    def _compute_surface_area(self):
        children = self.children
        diameter = self.diameter
        if self.is_sphere():
            area = surface_area_sphere(diameter)
        else:
            area = surface_area_cylinder(diameter, self._secondary_length())
            # Account for the surface area on the tips of terminal/leaf segments.
            if len(children) == 0:
                area += surface_area_disk(diameter)
        # Account for the surface area covered by children.
        for child in children:
            if not child._primary:
                attachment_diameter = min(diameter, child.diameter)
                area -= surface_area_disk(attachment_diameter)
        self.surface_area = area

    @Compute
    def _compute_cross_sectional_area(self):
        self.cross_sectional_area = surface_area_disk(self.diameter)

    @Compute
    def _compute_intracellular_volume(self):
        if self.is_sphere():
            self.volume = volume_sphere(self.diameter)
        else:
            self.volume = volume_cylinder(self.diameter, self._secondary_length())
