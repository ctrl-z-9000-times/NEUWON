from neuwon.database import *
import pytest
import random

def test_connectivity_matrix():
    # Make objects which are connected to each other with random connectivity.
    db = Database()
    Foo = db.add_class("Foo")
    Foo.add_connectivity_matrix("f", Foo)

    num = 1000
    Foo = Foo.get_instance_type()
    all_foo = [Foo() for _ in range(num)]
    for q in all_foo:
        q.f = random.sample(all_foo, 2)

    # Do a random walk through the graph, using the OOP API.
    cursor = start = random.choice(all_foo)
    path_history = []
    for _ in range(20):
        cursor = random.choice(cursor.f)
        path_history.append(cursor)

    # Use the connectivity matrix to check that the cursor's path was a valid
    # path through the graph.
    possible = np.zeros(num)
    possible[start.get_unstable_index()] = 1
    x = db.get("Foo.f").to_csr().get_data()
    for cursor in path_history:
        possible = possible * x
        assert possible[cursor.get_unstable_index()] > 0
