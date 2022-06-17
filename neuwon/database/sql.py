""" Serialize the database with SQL. """

from .data_components import *
from .dtypes import *
import numpy as np
import sqlite3
import struct

def schema(database) -> str:
    tables = []
    # Create tables for every class to store its attributes.
    for db_class in database.get_all_classes():
        columns = []
        for db_component in db_class.get_all_components():
            if not isinstance(db_component, Attribute):
                continue
            dtype = db_component.dtype
            if db_component.is_reference():
                raise NotImplementedError
            elif dtype == np.float32:   sql_dtype = 'FLOAT'
            elif dtype == np.float64:   sql_dtype = 'REAL'
            elif dtype == np.bool_:     sql_dtype = 'BOOLEAN'
            elif dtype == np.int8:      sql_dtype = 'SMALLINT'
            elif dtype == np.uint8:     sql_dtype = 'SMALLINT'
            elif dtype == np.int16:     sql_dtype = 'MEDIUMINT'
            elif dtype == np.uint16:    sql_dtype = 'MEDIUMINT'
            elif dtype == np.int32:     sql_dtype = 'INT'
            elif dtype == np.uint32:    sql_dtype = 'INT'
            elif dtype == np.int64:     sql_dtype = 'BIGINT'
            elif dtype == np.uint64:    sql_dtype = 'BIGINT'
            else: raise NotImplementedError(dtype)
            columns.append(f'{db_component.get_name()} {sql_dtype}')
        columns = ', '.join(columns)
        tables.append(f'CREATE TABLE {db_class.get_name()} ({columns});')
    # Create one table for all class attributes.
    for db_class in database.get_all_classes():
        for db_component in db_class.get_all_components():
            if not isinstance(db_component, ClassAttribute):
                continue
            raise NotImplementedError
    # TODO: How to deal with sparse matrices?
    for db_class in database.get_all_classes():
        for db_component in db_class.get_all_components():
            if not isinstance(db_component, SparseMatrix):
                continue
            raise NotImplementedError
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
                data = cur.fetchall()
                dtype = component.get_dtype()

                # print(component.name, dtype, data)

                if dtype.kind == 'u':
                    data = [int.from_bytes(x, 'little', signed=False) for (x,) in data]
                elif dtype.kind == 'i':
                    data = [int.from_bytes(x, 'little', signed=True) for (x,) in data]
                elif dtype == np.float32:
                    data = [struct.unpack('f', x) for (x,) in data]

                data = np.array(data, dtype=component.get_dtype())
                component.get_data()[idx] = data.flatten()

    con.close()

