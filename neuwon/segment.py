"""
Segments are organized in a tree.
The root of the tree is a sphere,
all other segments are cylinders.
"""

import numpy as np
from neuwon.database import epsilon
import re

class Tree:
    @classmethod
    def _initialize(cls, database):
        db_cls = database.add_class("Segment", cls)
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

    def is_root(self):
        return self.parent is None

class Geometry(Tree):
    @classmethod
    def _initialize(cls, database):
        super()._initialize(database)
        db_cls = database.get_class("Segment")
        db_cls.add_attribute("coordinates", shape=(3,), units="μm")
        db_cls.add_attribute("diameter", valid_range=(0.0, np.inf), units="μm")
        db_cls.add_attribute("_primary", dtype=np.bool, doc="""
                Primary segments are straightforward extensions of the parent
                branch. Non-primary segments are lateral branches off to the side of
                the parent branch.  """)
        db_cls.add_attribute("length", units="μm", doc="""
                The distance between this node and its parent node.
                Root node lengths are their radius.\n""")
        db_cls.add_attribute("surface_area", valid_range=(epsilon, np.inf), units="μm²")
        db_cls.add_attribute("cross_sectional_area", units="μm²",
                valid_range = (epsilon, np.inf))
        db_cls.add_attribute("volume", valid_range=(epsilon * (1e-6)**3, np.inf), units="Liters")

    def __init__(self, parent, coordinates, diameter):
        super().__init__(parent)
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

    def is_sphere(self):
        return self.is_root()

    def is_cylinder(self):
        return not self.is_sphere()

    def _compute_length(self):
        parent = self.parent
        if self.is_sphere():
            # Spheres have no defined length, so make one up instead instead.
            self.length = (2/3) * self.diameter
        else:
            length = np.linalg.norm(self.coordinates - parent.coordinates)
            # Subtract the parent's radius from the secondary nodes length,
            # to avoid excessive overlap between segments.
            if not self._primary:
                parent_radius = 0.5 * parent.diameter
                if length < parent_radius + epsilon:
                    # This segment is entirely enveloped within its parent. In
                    # this corner case allow the segment to protrude directly
                    # from the center of the parent instead of the surface.
                    pass
                else:
                    length -= parent_radius
            self.length = length

    def _compute_surface_area(self):
        children = self.children
        diameter = self.diameter
        if self.is_sphere():
            surface_area = _surface_area_sphere(diameter)
        else:
            surface_area = _surface_area_cylinder(diameter, self.length)
            # Account for the surface area on the tips of terminal/leaf segments.
            if len(children) == 0:
                surface_area += _area_circle(diameter)
        # Account for the surface area covered by children.
        for child in children:
            if not child._primary:
                attachment_diameter = min(diameter, child.diameter)
                surface_area -= _area_circle(attachment_diameter)
        self.surface_area = surface_area

    def _compute_cross_sectional_area(self):
        self.cross_sectional_area = _area_circle(self.diameter)

    def _compute_intracellular_volume(self):
        if self.is_sphere():
            self.volume = 1000 * (4/3) * np.pi * (self.diameter/2) ** 3
        else:
            self.volume = 1000 * np.pi * (self.diameter/2) ** 2 * self.length

    @classmethod
    def load_swc(cls, swc_data):
        # TODO: Arguments for coordinate offsets and rotations.
        swc_data = str(swc_data)
        if swc_data.endswith(".swc"):
            with open(swc_data, 'rt') as f:
                swc_data = f.read()
        swc_data = re.sub(r"#.*", "", swc_data) # Remove comments.
        swc_data = [x.strip() for x in swc_data.split("\n") if x.strip()]
        entries = dict()
        for line in swc_data:
            cursor = iter(line.split())
            sample_number = int(next(cursor))
            structure_id = int(next(cursor))
            coords = (float(next(cursor)), float(next(cursor)), float(next(cursor)))
            radius = float(next(cursor))
            parent = int(next(cursor))
            entries[sample_number] = cls(entries.get(parent, None), coords, 2 * radius)

class Electrics(Geometry):
    @classmethod
    def _initialize(cls, database,
                initial_voltage = -70,
                cytoplasmic_resistance = 1,
                membrane_capacitance = .01,):
        super()._initialize(database)
        db_cls = database.get_class("Segment")
        db_cls.add_attribute("voltage", initial_value=float(initial_voltage), units="mV")
        db_cls.add_attribute("axial_resistance", units="")
        db_cls.add_attribute("capacitance", units="Farads", valid_range=(0, np.inf))
        db_cls.add_class_attribute("cytoplasmic_resistance", cytoplasmic_resistance,
                units="?",
                valid_range=(epsilon, np.inf))
        db_cls.add_class_attribute("membrane_capacitance", membrane_capacitance,
                units="?",
                valid_range=(epsilon, np.inf))
        db_cls.add_attribute("_sum_conductances", units="Siemens", valid_range=(0, np.inf))
        db_cls.add_attribute("_driving_voltage", units="mV")
        # db.add_linear_system("membrane/diffusion", function=_electric_coefficients, epsilon=epsilon)

    def __init__(self, parent, coordinates, diameter):
        super().__init__(parent, coordinates, diameter)
        self._compute_passive_electric_properties()

    def _compute_passive_electric_properties(self):
        Ra = self.cytoplasmic_resistance
        Cm = self.membrane_capacitance

        # Compute axial membrane resistance.
        # TODO: This formula only works for cylinders.
        self.axial_resistance = Ra * self.length / self.cross_sectional_area
        # Compute membrane capacitance.
        self.capacitance = Cm * self.surface_area

        # TODO: Currently, diffusion from non-primary branches omits the section
        # of diffusive material between the parents surface and center.
        # Model the parents side of intracellular volume as a frustum to diffuse through.
        # Implementation:
        # -> Directly include this frustum into the resistance calculations.
        # -> For chemical diffusion: make a new attribute for the geometry terms
        #    in the diffusion equation, and include the new frustum in the new attribute.

    @staticmethod
    def _electric_coefficients(access):
        """
        Model the electric currents over the membrane surface (in the axial directions).
        Compute the coefficients of the derivative function:
        dV/dt = C * V, where C is Coefficients matrix and V is voltage vector.
        """
        dt           = access("time_step") / 1000 / _ITERATIONS_PER_TIMESTEP
        parents      = access("membrane/parents").get()
        resistances  = access("membrane/axial_resistances").get()
        capacitances = access("membrane/capacitances").get()
        src = []; dst = []; coef = []
        for child, parent in enumerate(parents):
            if parent == NULL: continue
            r        = resistances[child]
            c_parent = capacitances[parent]
            c_child  = capacitances[child]
            src.append(child)
            dst.append(parent)
            coef.append(+dt / (r * c_parent))
            src.append(child)
            dst.append(child)
            coef.append(-dt / (r * c_child))
            src.append(parent)
            dst.append(child)
            coef.append(+dt / (r * c_child))
            src.append(parent)
            dst.append(parent)
            coef.append(-dt / (r * c_parent))
        return (coef, (dst, src))

    def inject_current(self, current, duration=1e-3):
        self.model._injected_currents.inject_current(self.entity.index, current, duration)

def _area_circle(diameter):
    return 0.25 * np.pi * (diameter ** 2)

def _volume_sphere(diameter):
    return 1/0

def _surface_area_sphere(diameter):
    return np.pi * (diameter ** 2)

def _surface_area_cylinder(diameter, length):
    return np.pi * diameter * length

def _surface_area_frustum(radius_1, radius_2, length):
    """ Lateral surface area, does not include the ends. """
    s = sqrt((radius_1 - radius_2) ** 2 + length ** 2)
    return np.pi * (radius_1 + radius_2) * s

def _volume_of_frustum(radius_1, radius_2, length):
    return np.pi / 3.0 * length * (radius_1 * radius_1 + radius_1 * radius_2 + radius_2 * radius_2)
