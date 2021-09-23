""" Hodgkin-Huxley Demonstration. """

import neuwon.model

min_v = -88.
max_v = +60.

def make_model_with_hh(time_step):
    m = neuwon.model.Model(time_step, celsius = 6.3)
    m.add_species("na", reversal_potential = +60)
    m.add_species("k",  reversal_potential = -88)
    m.add_species("l",  reversal_potential = -54.3,)
    m.add_reaction("./nmodl_library/hh.mod")
    return m

if __name__ == '__main__':
    from neuwon.examples.HH.__main__ import main
    main()
