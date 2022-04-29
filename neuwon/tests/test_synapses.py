import neuwon
import pytest

@pytest.mark.skip()
def test_basic_mechanisms():
    # Make a single very simple synapse and verify approximate behavior.
    # Do NOT test the growth routines, manual setup.
    m = neuwon.Model(
        species = {
            'glu': {
                'outside': {
                    'initial_concentration': 0,
                    'decay_period': 1,
                },
            },
            'zero': {'reversal_potential': 0},
            'k':    {'reversal_potential': -70},
        },
        mechanisms = {
            'glu_presyn':   './neuwon/tests/mechanisms/glu_presyn.mod',
            'ampa':         './neuwon/tests/mechanisms/ampa.mod',
            'leak':         './neuwon/tests/mechanisms/leak.mod',
        },
        synapses = {
            'MySyn': {
                'cleft': {
                    'volume': 0.01,
                },
                'attachment_points': (
                    {
                        'mechanisms': {
                            'glu_presyn',
                        },
                    },
                    {
                        'mechanisms': {
                            'ampa',
                        },
                    },
                ),
            },
        },
    )
    Neuron = m.get_Neuron()
    Leak   = m.mechanisms['leak']
    MySyn  = m.synapses['MySyn']

    s1 = Neuron([0,0,0], 10).root
    s2 = Neuron([22,0,0], 10).root
    Leak(s1)
    Leak(s2)
    syn = MySyn(s1, s2)
    m.check()

    help(MySyn)

    # Run with no input.
    while m.clock() < 2:
        m.advance()
    assert s1.voltage == pytest.approx(-70)
    assert s2.voltage == pytest.approx(-70)
    assert syn.cleft.glu == 0

    # Run with presynaptic AP and verify postsynaptic EPSP.
    m.clock.reset()
    neuwon.TimeSeries().constant_wave(20, 2).play(s1, 'voltage', mode='=')
    while m.clock() < 4:
        m.advance()
    assert syn.cleft.glu > 0
    assert s2.voltage > -65

    # Run for a long time and verify EPSP is gone.
    m.clock.reset()
    while m.clock() < 10:
        m.advance()
    assert s2.voltage < -65

    m.check()


def test_network_ei():
    # I want this to be a whole suite of tests in itself...
    # There are a lot of interesting properties of E/I networks that I can easily verify.
    #       -> Use only point neurons w/o topology.

    model = neuwon.Model(
        species = {
            'glu': {
                'outside': {
                    'initial_concentration': 0,
                    'decay_period': .5,
            },},
            'gaba': {
                'outside': {
                    'initial_concentration': 0,
                    'decay_period': 2,
            },},
            'zero': {'reversal_potential': 0},
            'na': {'reversal_potential': +60},
            'k':  {'reversal_potential': -80},
        },
        mechanisms = {
            'hh':               './neuwon/tests/mechanisms/hh.mod',
            'glu_presyn':       './neuwon/tests/mechanisms/glu_presyn.mod',
            'gaba_presyn':      './neuwon/tests/mechanisms/gaba_presyn.mod',
            'ampa':             './nmodl_library/AMPA.mod',
            'nmda':             './nmodl_library/NMDA.mod',
            'gaba_receptor':    './neuwon/tests/mechanisms/gaba_receptor.mod',
            'leak':             './neuwon/tests/mechanisms/leak.mod',
        },
        regions = {
            'main': ('Rectangle', [0,0,0], [200,200,200]),
        },
        neurons = {
            'excit': ({
                    'segment_type': 'excit',
                    'region': 'main',
                    'number': 50,
                    'diameter': 10,
                    'mechanisms': {
                        'hh',
                    },
            },),
            'inhib': ({
                    'segment_type': 'inhib',
                    'region': 'main',
                    'number': 40,
                    'diameter': 10,
                    'mechanisms': {
                        'hh',
                    },
            },),
        },
        synapses = {
            'excit_syn': {
                'number': 50 * 50,
                'cleft': {
                    'volume': 0.01,
                },
                'attachment_points': (
                    {
                        'constraints': {
                            'segment_types': ('excit',),
                        },
                        'mechanisms': {
                            'glu_presyn': 0.1,
                    },},
                    {
                        'mechanisms': {
                            'ampa',
                    },},
            ),},
            'inhib_syn': {
                'number': 40 * 25,
                'maximum_distance': 50,
                'cleft': {
                    'volume': 0.01,
                },
                'attachment_points': (
                    {
                        'constraints': {
                            'segment_types': ('inhib',),
                        },
                        'mechanisms': {
                            'gaba_presyn',
                    },},
                    {
                        'mechanisms': {
                            'gaba_receptor',
                    },},
            ),},
    },)

    hh = model.mechanisms['hh']
    print(hh._advance_pycode)

    max_v = 61
    min_v = -81

    import random
    from neuwon.gui.viewport import Viewport
    # view = Viewport(camera_position=[0,0,400])
    # view.set_scene(model)
    voltage = model.Segment.get_database_class().get("voltage")

    excit = model.filter_segments_by_type(segment_types=['excit'])
    inhib = model.filter_segments_by_type(segment_types=['inhib'])
    next_stim = 0

    data = neuwon.TimeSeries.record_many(excit, 'voltage')

    while model.clock() < 500:
        if model.clock() >= next_stim:
            n = random.choice(excit)
            n.inject_current(1e-9, 1)
            next_stim += 100

        # v = ((voltage.get_data() - min_v) / (max_v - min_v)).clip(0, 1)
        # colors = [(x, 0, 1-x) for x in v]
        # view.tick(colors)
        model.advance()
        # model.check()

    neuwon.TimeSeries.plot_many(data, 100, color='k')


    1/0

    # transient input/output regime.
    # sustained recurrent activity regime.
    # run away excitation regime (kill all of the gaba neurons).

    # IDEA: Start with an over inhibited network, and kill progressively more
    # gaba neurons and observe the effects. It should have a very large range
    # of stable configurations.
    #   -> Measure the gain as a function of the E/I ratio?

