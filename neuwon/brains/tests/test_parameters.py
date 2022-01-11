from neuwon.brains.parameters import Parameters

test_parameters = {
    'foo': 4,
    'bar': (3,4,5.1),
    'complex': (2+2j),
    'strings': "hello",
    'struct': {
        'nested': {'foo': 55.0},
        'bar': 66,
    },
}

def test_construct():
    p = Parameters(test_parameters)
    assert p['strings'] == 'hello'
    assert p['struct']['nested']['foo'] == 55
    s = repr(p)
    assert 'hello' in s

def test_defaults():
    p = Parameters(test_parameters)
    p.update_with_defaults({
        'foo': -1,
        'foobar': -2,
        'struct': {
            'nested': {
                'foo': -1,
                'foobar': -2,
            }
        }
    })
    assert p['foo'] == 4 # Had existing value, should not overwrite.
    assert p['foobar'] == -2
    assert p['struct']['nested']['foo'] == 55 # Had existing value, should not overwrite.
    assert p['struct']['nested']['foobar'] == -2
