from neuwon.regions import RegionFactory

test_parameters = {
    'foo': ('Sphere', (1,2,3), 4),
    'bar': ('Intersection', 'foo', ('Rectangle', (-1,-1,-1), (1,1,1)))
}

def test_region_factory():
    r = RegionFactory(test_parameters)
    assert r['foo'].radius == 4
    assert r['foo'].contains([1.1, 1.1, 1.1])
    assert not r['bar'].contains([1.1, 1.1, 1.1])
