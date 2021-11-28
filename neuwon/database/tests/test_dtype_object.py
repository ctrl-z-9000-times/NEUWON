from neuwon.database import Database

def test_dtype_object():
    db = Database()
    foo_data = db.add_class('Foo')
    foo_data.add_attribute('bar', dtype=object)
    foo_data.add_attribute('bar_w_default', 1234, dtype=object)

    # Make an instance of Foo.
    Foo = foo_data.get_instance_type()
    my_foo = Foo()
    assert my_foo.bar is None
    assert my_foo.bar_w_default == 1234

    # Test that this accepts any python object type.
    my_foo.bar = 5.67
    my_foo.bar = 5
    my_foo.bar = type
    my_foo.bar = lambda: 42
    my_foo.bar = 'hello'
    my_foo.bar = object()
    class MyCustomClass:
        pass
    my_foo.bar = value = MyCustomClass()

    # Test on all memory spaces. If the memory space doesn't support python
    # objects then this should silently leave the data on the host.
    for device in ('host', 'cuda'):
        with db.using_memory_space(device):
            assert my_foo.bar is value
