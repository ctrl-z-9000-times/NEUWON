from neuwon.database import *
from neuwon.database.memory_spaces import *
import pytest
import random

# TODO: Test multi-level sorts with references between them.

# TODO: Test on a all memory spaces.

is_sorted = lambda data: all(a <= b for a,b in zip(data, data[1:]))

def test_sort():
    db = Database()
    Foo_data = db.add_class("Foo", sort_key='bar')
    bar_data = Foo_data.add_attribute("bar", 42)
    Foo = Foo_data.get_instance_type()
    for _ in range(10): f = Foo(bar=random.random())

    assert not db.is_sorted()
    db.sort()
    assert db.is_sorted()
    assert is_sorted(bar_data.get_data())

    destroyed_value = f.bar
    print("Destroying", f)
    f.destroy()
    assert not db.is_sorted()

    instances = Foo_data.get_all_instances()
    print("Remaining", instances)
    assert f not in instances
    assert destroyed_value not in set(x.bar for x in instances)

    db.sort()
    assert is_sorted(bar_data.get_data())
    assert destroyed_value not in bar_data.get_data()

    ans = Foo()
    q = [Foo(bar=random.random()) for q in range(10)]
    q[3].destroy()
    q[7].destroy()
    q[6].destroy()
    db.sort()
    assert is_sorted(bar_data.get_data())
    assert ans.get_unstable_index() == (10-1+10-3)
