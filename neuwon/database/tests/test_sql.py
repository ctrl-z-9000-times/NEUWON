from neuwon.database import Database
import numpy as np
import pytest
import tempfile

def make_model(db):
    Foobar = db.add_class("Foobar")
    Foobar.add_attribute("x", initial_distribution=('uniform', 0, 1))
    
    # TODO: Test other dtypes
    Foobar.add_attribute("y", dtype=np.int32, initial_distribution=('uniform', 0, 10))

    # TODO: Test class attributes
    Foobar.add_class_attribute("A", 42)

    # TODO: Test sparse matrixes


    return Foobar.get_instance_type()

def test_sql():
    db = Database()
    Foobar = make_model(db)

    db.save_sqlite3('foobar.db')
    db.load_sqlite3('foobar.db')

    db = Database()
    Foobar = make_model(db)
    db.load_sqlite3('foobar.db')

    for _ in range(10):
        Foobar()

    db.save_sqlite3('foobar.db')

    db2 = Database()
    Foobar = make_model(db2)
    db2.load_sqlite3('foobar.db')
    assert len(Foobar.get_database_class()) == 10
    assert all(db.get_data('Foobar.x') == db2.get_data('Foobar.x'))
    db2.load_sqlite3('foobar.db')
    assert len(Foobar.get_database_class()) == 20

