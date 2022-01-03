""" Hodgkin-Huxley Demonstration. """

from neuwon.rxd.rxd_model import RxD_Model

min_v = -88.
max_v = +60.

def make_model_with_hh(time_step):
    return RxD_Model(time_step,
        celsius = 6.3,
        species = {
            'na': {'reversal_potential': +60,},
            'k': {'reversal_potential': -88,},
            'l': {'reversal_potential': -54.3,},
        },
        mechanisms = {
            'hh': './nmodl_library/hh.mod'
        },
    )
