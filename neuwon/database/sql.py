""" Serialize the database with SQL. """

from .data_components import *
from .dtypes import *
import numpy as np
import sqlite3

def schema(database) -> str:
    tables = []
    for db_class in database.get_all_classes():
        components = []
        for db_component in db_class.get_all_components():
            if isinstance(db_component, Attribute):
                pass
            elif isinstance(db_component, ClassAttribute):
                pass
            elif isinstance(db_component, SparseMatrix):
                pass
            dtype = db_component.dtype
            if   dtype == np.float32:   sql_dtype = 'FLOAT'
            elif dtype == np.float64:   sql_dtype = 'REAL'
            elif dtype == Pointer:      sql_dtype = 'INT'
            else: raise NotImplementedError(dtype)
            components.append(f'{db_component.get_name()} {sql_dtype}')
        components = ', '.join(components)
        tables.append(f'CREATE TABLE {db_class.get_name()} ({components});')
    return tables

def save_sqlite3(database, filename):
    database.sort()
    open(filename, 'w').close() # Truncate the file.
    con = sqlite3.connect(filename)
    cur = con.cursor()
    cur.executescript('\n'.join(schema(database)))

    for db_class in database.get_all_classes():
        attributes   = [x for x in db_class.get_all_components() if isinstance(x, Attribute)]
        attrs_list   = ', '.join(x.get_name() for x in attributes)
        placeholders = ', '.join(['?'] * len(attributes))
        sql = f"INSERT INTO {db_class.get_name()}({attrs_list}) VALUES ({placeholders})"
        for inst in db_class.get_all_instances():
            attrs_values = tuple(getattr(inst, x.get_name()) for x in attributes)
            cur.execute(sql, attrs_values);

    con.commit()
    con.close()

def load_sqlite3(database, filename):
    con = sqlite3.connect(filename)
    cur = con.cursor()

    for db_class in database.get_all_classes():
        cur = cur.execute(f'SELECT COUNT(*) FROM {db_class.get_name()};')
        num = cur.fetchone()[0]
        idx = db_class._init_many(num)
        for component in db_class.get_all_components():
            if isinstance(component, Attribute):
                cur.execute(f'SELECT {component.get_name()} FROM {db_class.get_name()}')
                component.get_data()[idx] = np.array(cur.fetchall()).flatten()

    con.close()

