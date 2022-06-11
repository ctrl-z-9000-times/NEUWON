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
            if isinstance(db_component, )
            dtype = db_component.dtype
            if   dtype == np.float32:   sql_dtype = 'FLOAT'
            elif dtype == np.float64:   sql_dtype = 'REAL'
            elif dtype == Pointer:      sql_dtype = 'INT'
            else: raise NotImplementedError(dtype)
            components.append(f'{db_component.get_name()} {sql_dtype}')
        tables.append(f'CREATE TABLE {db_class.get_name()} ({', '.join(components)});')
    return tables

def new_sqlite3(filename):
    # If file already exists, squak?
    con = sqlite3.connect(filename)
    return con.cursor()

def save_sqlite3(database, filename):
    cur = new_sqlite3(filename)
    cur.executescript('\n'.join(schema(database)))
    
    1/0

def load_sqlite3(database, filename):
    1/0
