import neuwon
import pytest

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
            'leak': {'reversal_potential': -70},
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


@pytest.mark.skip()
def test_network_ei():
    # I want this to be a whole suite of tests in itself...
    # There are a lot of interesting properties of E/I networks that I can easily verify.
    #       -> Use only point neurons w/o topology.
    1/0
