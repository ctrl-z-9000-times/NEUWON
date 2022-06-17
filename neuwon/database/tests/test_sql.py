from neuwon.database import Database
import numpy as np
import os
import pytest
import tempfile

all_dtypes = {
        'a': np.dtype('f4'),
        'b': np.dtype('f8'),
        'c': np.dtype('i1'),
        'd': np.dtype('u1'),
        'e': np.dtype('i2'),
        'f': np.dtype('u2'),
        'g': np.dtype('i4'),
        'h': np.dtype('u4'),
        'i': np.dtype('i8'),
        'j': np.dtype('u8'),
        'h': np.dtype('b'),
}

def make_model(db):
    Foobar = db.add_class("Foobar")
    for attr_name, dtype in all_dtypes.items():
        if 'f' in dtype.kind:
            min_ = -1.0
            max_ = +1.0
        else:
            min_ = np.iinfo(dtype).min
            max_ = np.iinfo(dtype).max
        Foobar.add_attribute(attr_name, dtype=dtype, initial_distribution=('uniform', min_, max_))


    # TODO: Test class attributes

    # Foobar.add_class_attribute("A", 42)

    # TODO: Test sparse matrixes


    return Foobar.get_instance_type()

def test_sql():
    db_filename = tempfile.NamedTemporaryFile(suffix='.db', delete=False).name

    db = Database()
    Foobar = make_model(db)
    db.save_sqlite(db_filename)
    db.load_sqlite(db_filename)
    # Re-create the same model and make sure it can load the saved data.
    db = Database()
    Foobar = make_model(db)
    db.load_sqlite(db_filename)
    # Add data to the database and save it.
    for _ in range(10):
        x = Foobar()
    db.save_sqlite(db_filename)
    # Re-create the same model and make sure it can load the saved data.
    db2 = Database()
    Foobar2 = make_model(db2)
    db2.load_sqlite(db_filename)
    assert len(Foobar.get_database_class()) == 10
    Foobar  = Foobar.get_database_class()
    Foobar2 = Foobar2.get_database_class()
    for name in all_dtypes.keys():
        assert all(Foobar.get_data(name) == Foobar2.get_data(name))
    # Check that loading appends to database, not overwriting it.
    db.load_sqlite(db_filename)
    assert len(Foobar) == 20
    db2.load_sqlite(db_filename)
    assert len(Foobar2) == 20

    os.remove(db_filename)

