from neuwon.database import Real, epsilon, Pointer, NULL
import numpy as np

class Extracellular:
    __slots__ = ()
    @staticmethod
    def _initialize(database,
                tortuosity = 1.55,
                maximum_distance = 20,):
        ecs_data = database.add_class(Extracellular)
        ecs_data.add_attribute('coordinates', shape=(3,), units='μm')
        ecs_data.add_attribute('volume', units='μm³')
        ecs_data.add_attribute('voxel', False, dtype=bool)
        ecs_data.add_class_attribute('tortuosity', tortuosity)

        ecs_data.add_sparse_matrix('neighbor_distances', Extracellular)
        ecs_data.add_sparse_matrix('neighbor_border_areas', Extracellular)
        ecs_data.add_sparse_matrix('xarea_over_distance', Extracellular)

        ecs_cls = ecs_data.get_instance_type()
        ecs_cls.maximum_distance = float(maximum_distance)
        ecs_cls._dirty = True

        segment_data = database.get_class('Segment')
        segment_data.add_attribute('outside', dtype=Extracellular, allow_invalid=True)

        return ecs_cls

    def __init__(self, coordinates, volume):
        self.coordinates = coordinates
        self.volume      = volume
        cls = type(self)
        cls._dirty = True

    def get_voxel(self, coordinates):
        # TODO: Return the voxel that contains the given coordinates.
        #       If it doesn't exist yet then create it.
        1/0

        # divide the coordinates by the voxel dimensions, round down to integer,
        # hash the vector of three integers. Keep a dict of hashes->voxels.

    @classmethod
    def _outside_diffusion_coefficients(cls, access):
        1/0 # TODO
        D = self.outside_diffusivity / cls.tortuosity ** 2
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
