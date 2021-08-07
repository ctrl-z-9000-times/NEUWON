
class Species:
    """ """

    # TODO: Consider getting rid of the standard library of species and mechanisms.
    # Instead provide it in code examples which the user can copy paste into their
    # code, or possible import directly from an "examples" sub-module (like with
    # htm.core: `import htm.examples`). The problem with this std-lib is that there
    # is no real consensus on what's standard? Species have a lot of arguments and
    # while there may be one scientifically correct value for each argument, the
    # user might want to omit options for run-speed. Mechanisms come in so many
    # different flavors too, with varying levels of bio-accuracy vs run-speed.
    # 
    # Also, modify add_reactions to accept a whole dictionary of reactions so
    # that this works: model.add_reactions(neuwon.examples.Hodgkin_Huxley.reactions)
    _library = {
        "na": {
            "charge": 1,
            "transmembrane": True,
            "reversal_potential": "nerst",
            "inside_concentration":   15,
            "outside_concentration": 145,
        },
        "k": {
            "charge": 1,
            "transmembrane": True,
            "reversal_potential": "nerst",
            "inside_concentration": 150,
            "outside_concentration":  4,
        },
        "ca": {
            "charge": 2,
            "transmembrane": True,
            "reversal_potential": "nerst",
            # "reversal_potential": "goldman_hodgkin_katz", # TODO: Does not work...
            "inside_concentration": 70e-6,
            "outside_concentration": 2,
            "inside_diffusivity": 1e-9,
        },
        "cl": {
            "charge": -1,
            "transmembrane": True,
            "reversal_potential": "nerst",
            "inside_concentration":   10,
            "outside_concentration": 110,
        },
        "glu": {
            "outside_diffusivity": 1e-9,
            "outside_decay_period": .5e-3,
        },
    }

    def __init__(self, name,
            charge = 0,
            transmembrane = False,
            reversal_potential = "nerst",
            # TODO: Consider allowing concentration=None, which would undefined
            # the concentration and remove the database entry. This way it does
            # not clutter up the DB schema documentation w/ unused junk.
            inside_concentration  = 0.0,
            outside_concentration = 0.0,
            inside_diffusivity    = None,
            outside_diffusivity   = None,
            inside_decay_period   = float("inf"),
            outside_decay_period  = float("inf"),
            use_shells = False,
            outside_grid = None,):
        """
        Arguments
        * inside_concentration:  initial value, units millimolar.
        * outside_concentration: initial value, units millimolar.
        * reversal_potential: is one of: number, "nerst", "goldman_hodgkin_katz"

        If diffusivity is not given, then the concentration is a global constant.
        """
        self.name = str(name)
        self.charge = int(charge)
        self.transmembrane = bool(transmembrane)
        try: self.reversal_potential = float(reversal_potential)
        except ValueError:
            self.reversal_potential = str(reversal_potential)
            assert(self.reversal_potential in ("nerst", "goldman_hodgkin_katz"))
        self.inside_concentration   = float(inside_concentration)
        self.outside_concentration  = float(outside_concentration)
        self.inside_global_const    = inside_diffusivity is None
        self.outside_global_const   = outside_diffusivity is None
        self.inside_diffusivity     = float(inside_diffusivity) if not self.inside_global_const else 0.0
        self.outside_diffusivity    = float(outside_diffusivity) if not self.outside_global_const else 0.0
        self.inside_decay_period    = float(inside_decay_period)
        self.outside_decay_period   = float(outside_decay_period)
        self.use_shells             = bool(use_shells)
        self.inside_archetype       = "inside" if self.use_shells else "membrane/inside"
        self.outside_grid           = tuple(float(x) for x in outside_grid) if outside_grid is not None else None
        assert(self.inside_concentration  >= 0.0)
        assert(self.outside_concentration >= 0.0)
        assert(self.inside_diffusivity    >= 0)
        assert(self.outside_diffusivity   >= 0)
        assert(self.inside_decay_period   > 0.0)
        assert(self.outside_decay_period  > 0.0)
        if self.inside_global_const:  assert self.inside_decay_period == np.inf
        if self.inside_global_const:  assert not self.use_shells
        if self.outside_global_const: assert self.outside_decay_period == np.inf

    def __repr__(self):
        return "neuwon.Species(%s)"%self.name

    def _initialize(self, database):
        db = database
        if self.inside_global_const:
            db.add_global_constant(self.inside_archetype+"/concentrations/" + self.name,
                    self.inside_concentration, units="millimolar")
        else:
            db.add_attribute(self.inside_archetype+"/concentrations/" + self.name,
                    initial_value=self.inside_concentration, units="millimolar")
            db.add_attribute(self.inside_archetype+"/delta_concentrations/" + self.name,
                    initial_value=0.0, units="millimolar / timestep")
            db.add_linear_system(self.inside_archetype+"/diffusions/" + self.name,
                    function=self._inside_diffusion_coefficients, epsilon=epsilon * 1e-9,)
        if self.outside_global_const:
            db.add_global_constant("outside/concentrations/" + self.name,
                    self.outside_concentration, units="millimolar")
        else:
            db.add_attribute("outside/concentrations/" + self.name,
                    initial_value=self.outside_concentration, units="millimolar")
            db.add_attribute("outside/delta_concentrations/" + self.name,
                    initial_value=0.0, units="millimolar / timestep")
            db.add_linear_system("outside/diffusions/" + self.name,
                    function=self._outside_diffusion_coefficients, epsilon=epsilon * 1e-9,)
        if self.transmembrane:
            db.add_attribute("membrane/conductances/" + self.name,
                    initial_value=0.0, bounds=(0, np.inf), units="Siemens")
            if isinstance(self.reversal_potential, float):
                db.add_global_constant("membrane/reversal_potentials/" + self.name,
                        self.reversal_potential, units="mV")
            elif (self.inside_global_const and self.outside_global_const
                    and self.reversal_potential == "nerst"):
                db.add_global_constant("membrane/reversal_potentials/" + self.name,
                        self._nerst_potential(self.charge, db.access("T"),
                                self.inside_concentration,
                                self.outside_concentration),
                        units="mV")
            else:
                db.add_attribute("membrane/reversal_potentials/" + self.name,
                        units="mV")

    def _reversal_potential(self, access):
        x = access("membrane/reversal_potentials/" + self.name)
        if isinstance(x, float): return x
        inside  = access(self.inside_archetype+"/concentrations/"+self.name)
        outside = access("outside/concentrations/"+self.name)
        if not isinstance(inside, float) and self.use_shells:
            inside = inside[access("membrane/inside")]
        if not isinstance(outside, float):
            outside = outside[access("membrane/outside")]
        T = access("T")
        if self.reversal_potential == "nerst":
            x[:] = self._nerst_potential(self.charge, T, inside, outside)
        elif self.reversal_potential == "goldman_hodgkin_katz":
            voltages = access("membrane/voltages")
            x[:] = self._goldman_hodgkin_katz(self.charge, T, inside, outside, voltages)
        else: raise NotImplementedError(self.reversal_potential)
        return x

    @staticmethod
    def _nerst_potential(charge, T, inside_concentration, outside_concentration):
        xp = cp.get_array_module(inside_concentration)
        ratio = xp.divide(outside_concentration, inside_concentration)
        return xp.nan_to_num(1e3 * R * T / F / charge * xp.log(ratio))

    @staticmethod
    def _goldman_hodgkin_katz(charge, T, inside_concentration, outside_concentration, voltages):
        xp = cp.get_array_module(inside_concentration)
        inside_concentration  = inside_concentration * 1e-3  # Convert from millimolar to molar
        outside_concentration = outside_concentration * 1e-3 # Convert from millimolar to molar
        z = (charge * F / (R * T)) * voltages
        return ((1e3 * charge * F) *
                (inside_concentration * Species._efun(-z) - outside_concentration * Species._efun(z)))

    @staticmethod
    @cp.fuse()
    def _efun(z):
        if abs(z) < 1e-4:
            return 1 - z / 2
        else:
            return z / (math.exp(z) - 1)

    def _inside_diffusion_coefficients(self, access):
        dt      = access("time_step") / 1000 / _ITERATIONS_PER_TIMESTEP
        parents = access("membrane/parents").get()
        lengths = access("membrane/lengths").get()
        xareas  = access("membrane/cross_sectional_areas").get()
        volumes = access("membrane/inside/volumes").get()
        if self.use_shells: raise NotImplementedError
        src = []; dst = []; coef = []
        for location in range(len(parents)):
            parent = parents[location]
            if parent == NULL: continue
            flux = self.inside_diffusivity * xareas[location] / lengths[location]
            src.append(location)
            dst.append(parent)
            coef.append(+dt * flux / volumes[parent])
            src.append(location)
            dst.append(location)
            coef.append(-dt * flux / volumes[location])
            src.append(parent)
            dst.append(location)
            coef.append(+dt * flux / volumes[location])
            src.append(parent)
            dst.append(parent)
            coef.append(-dt * flux / volumes[parent])
        for location in range(len(parents)):
            src.append(location)
            dst.append(location)
            coef.append(-dt / self.inside_decay_period)
        return (coef, (dst, src))

    def _outside_diffusion_coefficients(self, access):
        extracellular_tortuosity = 1.4 # TODO: FIXME: put this one back in the db?
        D = self.outside_diffusivity / extracellular_tortuosity ** 2
        dt          = access("time_step") / 1000 / _ITERATIONS_PER_TIMESTEP
        decay       = -dt / self.outside_decay_period
        recip_vol   = (1.0 / access("outside/volumes")).get()
        area        = access("outside/neighbor_border_areas")
        dist        = access("outside/neighbor_distances")
        flux_data   = D * area.data / dist.data
        src         = np.empty(2*len(flux_data))
        dst         = np.empty(2*len(flux_data))
        coef        = np.empty(2*len(flux_data))
        write_idx   = 0
        for location in range(len(recip_vol)):
            for ii in range(area.indptr[location], area.indptr[location+1]):
                neighbor = area.indices[ii]
                flux     = flux_data[ii]
                src[write_idx] = location
                dst[write_idx] = neighbor
                coef[write_idx] = +dt * flux * recip_vol[neighbor]
                write_idx += 1
                src[write_idx] = location
                dst[write_idx] = location
                coef[write_idx] = -dt * flux * recip_vol[location] + decay
                write_idx += 1
        return (coef, (dst, src))



class InsideMethods:
    @staticmethod
    def _initialize(db):
        db.add_archetype("inside", doc="Intracellular space.")
        db.add_attribute("membrane/inside", dtype="inside", doc="""
                A reference to the outermost shell.
                The shells and the innermost core are allocated in a contiguous block
                with this referencing the start of range of length "membrane/shells" + 1.
                """)
        db.add_attribute("membrane/shells", dtype=np.uint8)
        db.add_attribute("inside/membrane", dtype="membrane")
        db.add_attribute("inside/shell_radius", units="μm")
        db.add_attribute("inside/volumes",
                # bounds=(epsilon * (1e-6)**3, None),
                allow_invalid=True,
                units="Liters")
        db.add_sparse_matrix("inside/neighbor_distances", "inside")
        db.add_sparse_matrix("inside/neighbor_border_areas", "inside")

class OutsideMethods:
    @staticmethod
    def _initialize(db):
        db.add_archetype("outside", doc="Extracellular space using a voronoi diagram.")
        db.add_attribute("membrane/outside", dtype="outside", doc="")
        db.add_attribute("outside/coordinates", shape=(3,), units="μm")
        db.add_kd_tree(  "outside/tree", "outside/coordinates")
        db.add_attribute("outside/volumes", units="Liters")
        db.add_sparse_matrix("outside/neighbor_distances", "outside")
        db.add_sparse_matrix("outside/neighbor_border_areas", "outside")


    def _initialize_outside(self, locations):
        self._initialize_outside_inner(locations)
        touched = set()
        for neighbors in self.db.access("outside/neighbor_distances")[locations]:
            touched.update(neighbors.indices)
        touched.difference_update(set(locations))
        self._initialize_outside_inner(list(touched))

    def _initialize_outside_inner(self, locations):
        # TODO: Consider https://en.wikipedia.org/wiki/Power_diagram
        coordinates     = self.db.access("outside/coordinates").get()
        tree            = self.db.access("outside/tree")
        write_neighbor_cols = []
        write_neighbor_dist = []
        write_neighbor_area = []
        for location in locations:
            coords = coordinates[location]
            potential_neighbors = tree.query_ball_point(coords, 2 * self.max_outside_radius)
            potential_neighbors.remove(location)
            volume, neighbors = neuwon.voronoi.voronoi_cell(location,
                    self.max_outside_radius, np.array(potential_neighbors, dtype=Pointer), coordinates)
            write_neighbor_cols.append(list(neighbors['location']))
            write_neighbor_dist.append(list(neighbors['distance']))
            write_neighbor_area.append(list(neighbors['border_surface_area']))
        self.db.access("outside/neighbor_distances",
                sparse_matrix_write=(locations, write_neighbor_cols, write_neighbor_dist))
        self.db.access("outside/neighbor_border_areas",
                sparse_matrix_write=(locations, write_neighbor_cols, write_neighbor_area))


