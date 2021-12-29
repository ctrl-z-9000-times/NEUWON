import neuwon.rxd.voronoi
import numpy as np

class Extracellular:
    """ Extracellular Space with a voronoi diagram. """
    __slots__ = ()
    @staticmethod
    def _initialize(database,
                tortuosity = 1.55,
                fh_space = 300e-10, # Frankenhaeuser Hodgkin Space, in Angstroms
                volume_fraction = .20,
                max_outside_radius = 20e-6,
                ):
        ec_data = database.add_class(Extracellular)
        ec_data.add_attribute('coordinates', shape=(3,), units='μm')
        ec_data.add_attribute('volumes', units='μm³')

        ec_data.add_sparse_matrix('neighbor_distances', Extracellular)
        ec_data.add_sparse_matrix('neighbor_border_areas', Extracellular)
        ec_data.add_sparse_matrix('xarea_over_distance', Extracellular)

        ec_cls = ec_data.get_instance_type()
        ec_cls._dirty = []
        ec_cls.kd_tree = None

        segment_data = database.get_class('Segment')
        segment_data.add_attribute('outside', dtype=Extracellular, allow_invalid=True)

        return ec_cls

    def __init__(self, coordinates, volume):
        self.coordinates = coordinates
        self.volume      = volume
        type(self)._dirty.append(self)

    def _clean(self):
        self._compute_voronoi_cells(self._dirty)
        touched = set()
        for neighbors in self.db.access("outside/neighbor_distances")[self._dirty]:
            touched.update(neighbors.indices)
        touched.difference_update(set(self._dirty))
        self._compute_voronoi_cells(list(touched))
        self._dirty.clear()

    @classmethod
    def _compute_voronoi_cells(cls, locations):
        1/0
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
            volume, neighbors = neuwon.species.voronoi.voronoi_cell(location,
                    self.max_outside_radius, np.array(potential_neighbors, dtype=Pointer), coordinates)
            write_neighbor_cols.append(list(neighbors['location']))
            write_neighbor_dist.append(list(neighbors['distance']))
            write_neighbor_area.append(list(neighbors['border_surface_area']))
        self.db.access("outside/neighbor_distances",
                sparse_matrix_write=(locations, write_neighbor_cols, write_neighbor_dist))
        self.db.access("outside/neighbor_border_areas",
                sparse_matrix_write=(locations, write_neighbor_cols, write_neighbor_area))

    @classmethod
    def _outside_diffusion_coefficients(cls, access):
        1/0
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
