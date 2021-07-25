from neuwon.database import *

class Foo:
    """ Equivalent OOP class definition. """
    def __init__(self):
        self.bar = 4

def test_OOP_API():
    class Foo(Instance):
        def my_helper_method(self):
            self.bar

    db = Database()
    Foo = db.add_class("Foo", instance_class=Foo)
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

    Foo.add_sparse_matrix("connections", "Foo").to_lil()
    print((x.connections))
    x.connections = ([y], [2.])
    print((x.connections))

    Foo.add_list_attribute("friends", dtype=Foo)
    x.friends.append(y)

    if False:
        help(Foo)
        help(x)
