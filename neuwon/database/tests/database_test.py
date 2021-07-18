from neuwon.database import *

def test_OOP_API():
    class InstanceClass:
        def my_helper_method(self):
            self.bar

    class Foo:
        def __init__(self):
            self.bar = 4

    db = Database()
    Foo = db.add_class("Foo", instance_class=InstanceClass)
    Foo.add_attribute("bar")

    x = Foo()
    x.bar = 4
    assert x.bar == 4

    x.my_helper_method()
    Foo.add_attribute("ptr", dtype="Foo")
    y = Foo()
    x.ptr = y
    x.ptr.bar = 5
    assert y.bar == 5
    Foo.add_class_attribute("shared_value", 77)
    assert x.shared_value == 77
    y.shared_value = 55
    assert x.shared_value == 55

    Foo.add_sparse_matrix("connections", "Foo").to_fmt('lil')
    print((x.connections))
    x.connections = ([y], [2.])
    print((x.connections))

    if False:
        help(Foo)
        help(x)

if __name__ == "__main__":
    test_OOP_API()
