from neuwon.database import Real, epsilon, Pointer, NULL
from neuwon.rxd.voronoi import voronoi_cell
from scipy.spatial import KDTree
import numpy as np

class Extracellular:
    """ Extracellular Space with a voronoi diagram. """
    __slots__ = ()
    @staticmethod
    def _initialize(database,
                tortuosity = 1.55,
                maximum_distance = 20,
                ):
        ecs_data = database.add_class(Extracellular)
        ecs_data.add_attribute('coordinates', shape=(3,), units='μm')
        ecs_data.add_attribute('voronoi_volume', units='μm³')
        ecs_data.add_attribute('volume', units='μm³')
        ecs_data.add_class_attribute('tortuosity', tortuosity)

        ecs_data.add_sparse_matrix('neighbor_distances', Extracellular)
        ecs_data.add_sparse_matrix('neighbor_border_areas', Extracellular)
        ecs_data.add_sparse_matrix('xarea_over_distance', Extracellular)

        ecs_cls = ecs_data.get_instance_type()
        ecs_cls.maximum_distance = float(maximum_distance)
        ecs_cls._dirty = []
        ecs_cls._kd_tree = None

        segment_data = database.get_class('Segment')
        segment_data.add_attribute('outside', dtype=Extracellular, allow_invalid=True)

        return ecs_cls

    def __init__(self, coordinates, volume):
        self.coordinates = coordinates
        self.volume      = volume
        cls = type(self)
        cls._dirty.append(self)
        cls._kd_tree = None

    @property
    def neighbors(self):
        return self.neighbor_distances[0]

    @classmethod
    def _clean(cls):
        if not cls._dirty: return
        db_class    = cls.get_database_class()
        assert db_class.get_database().is_sorted()
        dirty = [ecs.get_unstable_index() for ecs in cls._dirty]
        coordinates = db_class.get_data("coordinates")
        cls._kd_tree = KDTree(coordinates)
        cls._compute_voronoi_cells(dirty)
        neighbors_matrix = db_class.get("neighbor_distances").to_lil().get_data()
        touched = set()
        for location in dirty:
            neighbors = neighbors_matrix.getrow(location)
            touched.update(neighbors.rows[0])
        touched.difference_update(dirty)
        cls._compute_voronoi_cells(list(touched))
        cls._dirty = []

    @classmethod
    def _compute_voronoi_cells(cls, locations):
        # TODO: Consider https://en.wikipedia.org/wiki/Power_diagram
        db_class        = cls.get_database_class()
        coordinates     = db_class.get_data("coordinates")
        voronoi_volume  = db_class.get_data("voronoi_volume")
        for location in locations:
            potential_neighbors = cls._kd_tree.query_ball_point(
                                    coordinates[location], 2 * cls.maximum_distance)
            potential_neighbors.remove(location)
            volume, neighbors = voronoi_cell(location,
                                            cls.maximum_distance,
                                            np.array(potential_neighbors, dtype=Pointer),
                                            coordinates)
            voronoi_volume[location] = volume
            neighbor_locations = list(neighbors['location'])
            neighbor_distances = list(neighbors['distance'])
            neighbor_areas     = list(neighbors['border_surface_area'])
            db_class.get("neighbor_distances").write_row(
                    location, neighbor_locations, neighbor_distances)
            db_class.get("neighbor_border_areas").write_row(
                    location, neighbor_locations, neighbor_areas)

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

    @classmethod
    def fill_region(cls, maximum_distance, region) -> list:
        raise NotImplementedError
