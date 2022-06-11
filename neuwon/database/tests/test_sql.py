from neuwon.database import Database
import pytest
import tempfile

def test_sql():
    db = Database()
    Foobar = db.add_class("Foobar")
    Foobar.add_attribute("x", 3)
    Foobar = Foobar.get_instance_type()

    db.save_sqlite3('foobar.db')
    db.load_sqlite3('foobar.db')

    for _ in range(10):
        Foobar()

    db.save_sqlite3('foobar.db')
    db.load_sqlite3('foobar.db')

    assert len(Foobar.get_database_class()) == 20

