import numpy as np

from neuwon.database import epsilon

class SegmentMethods:
    """

    Segments are organized in a tree.
    The root of the tree is a sphere,
    all other segments are cylinders.
    """
    @classmethod
    def _make_Segment_class(cls, db,
                initial_voltage = -70,
                cytoplasmic_resistance = 1,
                membrane_capacitance = .01,):
        cls = db.add_class("Segment", cls)
        cls.add_attribute("parent", dtype=cls, allow_invalid=True)
        cls.add_connectivity_matrix("children", cls)
        cls.add_attribute("coordinates", shape=(3,), units="μm")
        cls.add_attribute("diameter", valid_range=(0.0, np.inf), units="μm")
        cls.add_attribute("_primary", dtype=np.bool, doc="""
                Primary segments are straightforward extensions of the parent
                branch. Non-primary segments are lateral branches off to the side of
                the parent branch.  """)
        cls.add_attribute("length", units="μm", doc="""
                The distance between this node and its parent node.
                Root node lengths are their radius.\n""")
        cls.add_attribute("surface_area", valid_range=(epsilon, np.inf), units="μm²")
        cls.add_attribute("cross_sectional_area", units="μm²",
                valid_range = (epsilon, np.inf))
        cls.add_attribute("volumes", valid_range=(epsilon * (1e-6)**3, np.inf), units="Liters")
        # Electic properties and internal variables.
        cls.add_attribute("voltage", initial_value=float(initial_voltage), units="mV")
        cls.add_attribute("axial_resistance", units="")
        cls.add_attribute("capacitance", units="Farads", valid_range=(0, np.inf))
        cls.add_class_attribute("cytoplasmic_resistance", cytoplasmic_resistance,
                units="?",
                valid_range=(epsilon, np.inf))
        cls.add_class_attribute("membrane_capacitance", membrane_capacitance,
                units="?",
                valid_range=(epsilon, np.inf))
        cls.add_attribute("_sum_conductances", units="Siemens", valid_range=(0, np.inf))
        cls.add_attribute("_driving_voltage", units="mV")
        # db.add_linear_system("membrane/diffusion", function=_electric_coefficients, epsilon=epsilon)

        return cls.get_instance_type()

    def __init__(self, parent, coordinates, diameter, **kwargs):
        super().__init__(**kwargs)
        self.parent = parent
        self.coordinates = coordinates
        self.diameter = diameter
        # Add ourselves to the parent's children list.
        parent = self.parent
        if parent is not None:
            siblings = parent.children
            siblings.append(self)
            parent.children = siblings
        # Determine _primary flag.
        sphere = parent is None # Root is sphere.
        if sphere:
            self._primary = False # Value does not matter.
        elif parent.parent is None: # Parent is root / sphere.
            self._primary = False # Spheres have no primary branches off of a them.
        else:
            # Set the first child added to a segment as the primary extension,
            # and all subsequent children as secondary branches.
            self._primary = len(siblings) < 2

        self._compute_geometry()
        # self._compute_passive_electric_properties()

    def _compute_geometry(self):
        parent = self.parent
        sphere = parent is None # Root is sphere.
        # Compute length.
        if sphere:
            # Spheres have no defined length, so use the radius instead.
            self.length = 0.5 * self.diameter
        else:
            distance = np.linalg.norm(self.coordinates - parent.coordinates)
            # Subtract the parent's radius from the secondary nodes length,
            # to avoid excessive overlap between segments.
            if not self._primary:
                parent_radius = 0.5 * parent.diameter
                if distance < parent_radius + epsilon:
                    # This segment is entirely enveloped within its parent. In
                    # this corner case allow the segment to protrude directly
                    # from the center of the parent instead of the surface.
                    pass
                else:
                    distance -= parent_radius
            self.length = distance
        # Compute surface areas.
        if sphere:
            self.surface_area = _surface_area_sphere(self.diameter)
        else:
            self.surface_area = _surface_area_cylinder(self.diameter, self.length)
            # Account for the surface area on the tips of terminal/leaf segments.
            # if self.children.getrow(idx).getnnz() == 0:
            #     self.surface_area += _area_circle(self.diameter)
        # Compute cross-sectional areas.
        self.cross_sectional_area = _area_circle(self.diameter)
        # Compute intracellular volumes.
        if sphere:
            self.volume = 1000 * (4/3) * np.pi * (self.diameter/2) ** 3
        else:
            self.volume = 1000 * np.pi * (self.diameter/2) ** 2 * self.length

    def _compute_passive_electric_properties(self):
        Ra = self.cytoplasmic_resistance
        Cm = self.membrane_capacitance

        # Compute axial membrane resistance.
        # TODO: This formula only works for cylinders.
        self.axial_resistance = Ra * lengths[membrane_idx] / x_areas[membrane_idx]
        # Compute membrane capacitance.
        self.capacitance = Cm * s_areas[membrane_idx]

        # TODO: Currently, diffusion from non-primary branches omits the section
        # of diffusive material between the parents surface and center.
        # Model the parents side of intracellular volume as a frustum to diffuse through.
        # Implementation:
        # -> Directly include this frustum into the resistance calculations.
        # -> For chemical diffusion: make a new attribute for the geometry terms
        #    in the diffusion equation, and include the new frustum in the new attribute.

        outside_volumes = access("outside/volumes")
        fh_space = self.fh_space * s_areas[membrane_idx] * 1000
        outside_volumes[access("membrane/outside")[membrane_idx]] = fh_space

    @staticmethod
    def _electric_coefficients(access):
        """
        Model the electric currents over the membrane surface.
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
