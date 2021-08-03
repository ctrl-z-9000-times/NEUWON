
class Segment(DB_Object):
    """ """
    @staticmethod
    def _initialize_database(db):
        cls = db.add_class("Segment")
        cls.add_attribute("parent", dtype=cls, allow_invalid=True)
        cls.add_connectivity_matrix("children", cls)
        cls.add_attribute("coordinates", shape=(3,), units="μm")
        cls.add_attribute("diameter", bounds=(0.0, None), units="μm")
        # cls.add_attribute("shapes", dtype=np.uint8, doc="""
        #         0 - Sphere
        #         1 - Cylinder

        #         Note: only and all root segments are spheres. """)
        # cls.add_attribute("primary", dtype=np.bool, doc="""
        #         Primary segments are straightforward extensions of the parent
        #         branch. Non-primary segments are lateral branches off to the side of
        #         the parent branch.  """)
        # cls.add_attribute("lengths", units="μm", doc="""
        #         The distance between each node and its parent node.
        #         Root node lengths are their radius.\n""")
        # cls.add_attribute("surface_areas", bounds=(epsilon, None), units="μm²")
        # cls.add_attribute("cross_sectional_areas", units="μm²",
        #         bounds = (epsilon, None))
        # cls.add_attribute("inside/volumes", bounds=(epsilon * (1e-6)**3, None), units="Liters")

        cls.add_attribute("voltages", initial_value=float(initial_voltage), units="mV")
        cls.add_attribute("axial_resistances", allow_invalid=True, units="")
        cls.add_attribute("capacitances", units="Farads", bounds=(0, None))
        cls.add_attribute("conductances", units="Siemens", bounds=(0, None))
        cls.add_attribute("driving_voltages", units="mV")
        # db.add_linear_system("membrane/diffusion", function=_electric_coefficients, epsilon=epsilon)

        return cls.get_instance_type()

    def __init__(self, parent, coordinates, diameter, shape, shells):
        1/0

    def _initialize_membrane_geometry(self):
        # Compute lengths.
        p = parents[idx]
        if shapes[idx] == 0: # Root sphere.
            lengths[idx] = 0.5 * diams[idx] # Spheres have no defined length, so use the radius.
        else:
            distance = np.linalg.norm(coords[idx] - coords[p])
            # Subtract the parent's radius from the secondary nodes length,
            # to avoid excessive overlap between segments.
            if not primary[idx]:
                parent_radius = diams[p] / 2.0
                if distance < parent_radius + epsilon * 1e-6:
                    # This segment is entirely enveloped within its parent. In
                    # this corner case allow the segment to protrude directly
                    # from the center of the parent instead of the surface.
                    pass
                else:
                    distance -= parent_radius
            lengths[idx] = distance
        # Compute surface areas.
        p = parents[idx]
        d = diams[idx]
        shape = shapes[idx]
        if shape == 0: # Sphere.
            s_areas[idx] = np.pi * (d ** 2)
        else:
            l = lengths[idx]
            if shape == 1: # Cylinder.
                s_areas[idx] = np.pi * d * l
            # Account for the surface area on the tips of terminal/leaf segments.
            if children.getrow(idx).getnnz() == 0:
                s_areas[idx] += _area_circle(d)
        # Compute cross-sectional areas.
        p = parents[idx]
        d = diams[idx]
        shape = shapes[idx]
        if shape == 0: # Sphere.
            x_areas[idx] = _area_circle(d)
        else:
            if shape == 1: # Cylinder.
                x_areas[idx] = _area_circle(d)
        # Compute intracellular volumes.
        p = parents[idx]
        d = diams[idx]
        l = lengths[idx]
        shape = shapes[idx]
        if shape == 0: # Sphere.
            volumes[idx] = 1000 * (4/3) * np.pi * (d/2) ** 3
        else:
            if shape == 1: # Cylinder.
                volumes[idx] = 1000 * np.pi * (d/2) ** 2 * l
        # Compute passive electric properties
        Ra       = self.cytoplasmic_resistance
        r        = access("membrane/axial_resistances")
        Cm       = self.membrane_capacitance
        c        = access("membrane/capacitances")
        # Compute axial membrane resistance.
        # TODO: This formula only works for cylinders.
        r[membrane_idx] = Ra * lengths[membrane_idx] / x_areas[membrane_idx]
        # Compute membrane capacitance.
        c[membrane_idx] = Cm * s_areas[membrane_idx]

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
