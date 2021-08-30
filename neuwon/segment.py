import numpy as np
from neuwon.database import epsilon
import re

__all__ = ["SegmentMethods"]

class Tree:
    """
    Segments are organized in a tree.
    """
    @staticmethod
    def _initialize(db_cls):
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

class Geometry:
    """
    The root of the tree is a sphere,
    all other segments are cylinders.
    """
    @staticmethod
    def _initialize(db_cls):
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

    def is_sphere(self):
        return self.is_root()

    def is_cylinder(self):
        return not self.is_sphere()

    def _compute_length(self):
        if self.is_sphere():
            # Spheres have no defined length, so make one up instead instead.
            self.length = (2.0 / 3.0) * self.diameter
        else:
            self.length = np.linalg.norm(self.coordinates - self.parent.coordinates)

    def _secondary_length(self):
        # Subtract the parent's radius from the secondary nodes length,
        # to avoid excessive overlap between segments.
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
            area = surface_area.sphere(diameter)
        else:
            area = surface_area.cylinder(diameter, self._secondary_length())
            # Account for the surface area on the tips of terminal/leaf segments.
            if len(children) == 0:
                area += surface_area.circle(diameter)
        # Account for the surface area covered by children.
        for child in children:
            if not child._primary:
                attachment_diameter = min(diameter, child.diameter)
                area -= surface_area.circle(attachment_diameter)
        self.surface_area = area

    def _compute_cross_sectional_area(self):
        self.cross_sectional_area = surface_area.circle(self.diameter)

    def _compute_intracellular_volume(self):
        if self.is_sphere():
            self.volume = volume.sphere(self.diameter)
        else:
            self.volume = volume.cylinder(self.diameter, self._secondary_length())

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

class ElectricProperties:
    @staticmethod
    def _initialize(db_cls, *,
                initial_voltage,
                cytoplasmic_resistance,
                membrane_capacitance,):
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

    def __init__(self):
        self._compute_passive_electric_properties()

    def _compute_passive_electric_properties(self):
        Ra = self.cytoplasmic_resistance
        Cm = self.membrane_capacitance

        # Compute axial membrane resistance.
        # TODO: This formula only works for cylinders.
        self.axial_resistance = Ra * self.length / self.cross_sectional_area
        # Compute membrane capacitance.
        self.capacitance = Cm * self.surface_area

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

    def inject_current(self, current, duration = 1.4):
        duration = float(duration)
        assert(duration >= 0)
        current = float(current)

        # Inject_Current is applied two times every tick, at the start of advance_species.
        # Need to make a second clock for the pre-species-advance stuff, put it in the model.
        # Put a link to the model (and via the model the second clock) in the
        # Segment class (where the _cls slot is).

        clock = 1/0
        dv = current * min(clocl.get_tick_period(), t) / self.capacitances
        TimeSeriesBuffer().set_data([dv, dv], [0,duration]).play(self, "voltage", clock=clock)

class SegmentMethods(Tree, Geometry, ElectricProperties):
    @classmethod
    def _initialize(cls, database,
                initial_voltage = -70,
                cytoplasmic_resistance = 1,
                membrane_capacitance = .01,):
        db_cls = database.add_class("Segment", cls)
        Tree._initialize(db_cls)
        Geometry._initialize(db_cls)
        ElectricProperties._initialize(db_cls,
                initial_voltage=initial_voltage,
                cytoplasmic_resistance=cytoplasmic_resistance,
                membrane_capacitance=membrane_capacitance,)

    def __init__(self, parent, coordinates, diameter):
        Tree.__init__(self, parent)
        Geometry.__init__(self, coordinates, diameter)
        ElectricProperties.__init__(self, )

class surface_area:
    # TODO: Consider renaming "circle" to "disk"?
    def circle(diameter):
        return np.pi * ((0.5 * diameter) ** 2)

    def sphere(diameter):
        return np.pi * (diameter ** 2)

    def cylinder(diameter, length):
        """ Lateral surface area, does not include the end caps. """
        return np.pi * diameter * length

    def frustum(radius_1, radius_2, length):
        """ Lateral surface area, does not include the end caps. """
        s = sqrt((radius_1 - radius_2) ** 2 + length ** 2)
        return np.pi * (radius_1 + radius_2) * s

class volume:
    def sphere(diameter):
        return (4.0 / 3.0 ) * np.pi * (0.5 * diameter) ** 3

    def cylinder(diameter, length):
        return surface_area.circle(diameter) * length

    def frustum(radius_1, radius_2, length):
        return np.pi / 3.0 * length * (radius_1 * radius_1 + radius_1 * radius_2 + radius_2 * radius_2)
