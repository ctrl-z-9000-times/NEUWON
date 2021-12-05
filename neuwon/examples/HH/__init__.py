""" Hodgkin-Huxley Demonstration. """

from neuwon.model import Model

min_v = -88.
max_v = +60.

def make_model_with_hh(time_step):
    return Model({
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
    )

if __name__ == '__main__':
    from neuwon.examples.HH.__main__ import main
    main()
