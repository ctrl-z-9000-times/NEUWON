""" Model of an action potential propagating through an axonal arbor. """

from neuwon import Model
from neuwon.gui.viewport import Viewport
from math import pi

def main(time_step = .1):
    model = Model({
            'time_step': time_step,
            'celsius': 6.3,
        },
        species = {
            'na': {'reversal_potential': +60,},
            'k': {'reversal_potential': -88,},
            'l': {'reversal_potential': -54.3,},
        },
        mechanisms = {
            'hh': './nmodl_library/hh.mod'
        },
        neurons = {
            'demo_neuron': ({
                    'segment_type': 'soma',
                    'region': ('Sphere', [0, 0, 0], 1),
                    'number': 1,
                    'diameter': 10,
                    'mechanisms': {'hh'}
                },
                {
                    'segment_type': 'axon',
                    'region': ('Sphere', [0, 0, 0], 500),
                    'diameter': .5,
                    'morphology': {
                        'carrier_point_density': 0.000025,
                        'balancing_factor': .0,
                        'extension_angle': pi / 6,
                        'extension_distance': 60,
                        'bifurcation_angle': pi / 3,
                        'bifurcation_distance': 40,
                        'extend_before_bifurcate': True,
                        'maximum_segment_length': 30,},
                    'mechanisms': {'hh'}
                },
            )
        },
    )
    max_v = +60
    min_v = -88
    soma  = model.Neuron.get_database_class().get_all_instances()[0].root

    if True:
        print("Number of Locations:", len(model))
        sa_units = soma.get_database_class().get("surface_area").get_units()
        print("Soma surface area:", soma.surface_area, sa_units)
        all_segments = model.Segment.get_database_class().get_all_instances()
        sa = sum(x.surface_area for x in all_segments)
        print("Total surface area:", sa, sa_units)

    view = Viewport(camera_position=[0,0,400])
    view.set_scene(model)
    voltage = model.Segment.get_database_class().get("voltage")

    period = 30
    cooldown = 0
    while True:
        if cooldown <= 0:
            soma.inject_current(2e-9, 1)
            cooldown = period / time_step
        else:
            cooldown -= 1
        v = ((voltage.get_data() - min_v) / (max_v - min_v)).clip(0, 1)
        colors = [(x, 0, 1-x) for x in v]
        view.tick(colors)
        model.advance()

if __name__ == "__main__":
    main()
