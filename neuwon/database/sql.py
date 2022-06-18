""" Serialize the database with SQL. """

from .data_components import *
from .dtypes import *
from collections.abc import Iterable
import numpy as np
import sqlite3
import struct

def _numpy_to_sql_dtype(dtype):
    if   dtype == np.bool_:     return 'BOOLEAN'
    elif dtype == np.int8:      return 'SMALLINT'
    elif dtype == np.uint8:     return 'SMALLINT'
    elif dtype == np.int16:     return 'MEDIUMINT'
    elif dtype == np.uint16:    return 'MEDIUMINT'
    elif dtype == np.int32:     return 'INT'
    elif dtype == np.uint32:    return 'INT'
    elif dtype == np.int64:     return 'BIGINT'
    elif dtype == np.uint64:    return 'BIGINT'
    elif dtype == np.float32:   return 'FLOAT'
    elif dtype == np.float64:   return 'REAL'
    else: raise NotImplementedError(dtype)

def _make_column(attribute):
    name        = attribute.get_name()
    sql_type    = _numpy_to_sql_dtype(attribute.get_dtype())
    column_spec = f'{name} {sql_type}'
    assert attribute.get_shape() == (1,) # TODO!
    if attribute.is_reference():
        column_spec += f' REFERENCES {attribute.reference}(_idx)'
        if not attribute.allow_invalid:
            column_spec += ' NOT NULL'
    else:
        default = attribute.get_initial_value()
        if default is not None and not isinstance(default, Iterable):
            column_spec += f' DEFAULT {default}'
        else:
            column_spec += ' NOT NULL'
        # 
        min_, max_ = attribute.get_valid_range()
        if min_ is not None and max_ is not None:
            column_spec += f' CHECK ({attr} >= {min_} & {attr} <= {max_})'
        elif min_ is not None:
            column_spec += f' CHECK ({attr} >= {min_})'
        elif max_ is not None:
            column_spec += f' CHECK ({attr} <= {max_})'
        if not attribute.allow_invalid:
            pass # TODO: Check for NaN's.
    return column_spec

def sqlite3_schema(database) -> str:
    tables = []
    # Create two tables for every class to store its instances and class-attributes.
    for db_class in database.get_all_classes():
        cls_name        = db_class.get_name()
        inst_columns    = [f'_idx {_numpy_to_sql_dtype(Pointer)} NOT NULL']
        cls_columns     = []
        for db_component in db_class.get_all_components():
            specification = _make_column(db_component)
            if isinstance(db_component, Attribute):
                inst_columns.append(specification)
            elif isinstance(db_component, ClassAttribute):
                cls_columns.append(specification)
        inst_columns.append('PRIMARY KEY (_idx)')
        inst_columns = ', '.join(inst_columns)
        tables.append(f'CREATE TABLE {cls_name} ({inst_columns});')
        if cls_columns:
            cls_columns = ', '.join(cls_columns)
            tables.append(f'CREATE TABLE {cls_name}_cls_attrs ({cls_columns});')
    # Create a join table for every sparse matrix.
    for db_class in database.get_all_classes():
        for db_component in db_class.get_all_components():
            if not isinstance(db_component, SparseMatrix):
                continue
            raise NotImplementedError # TODO: matrices
    return '\n'.join(tables)

def sqlite3_save(database, filename):
    database.sort()
    open(filename, 'w').close() # Truncate the file.
    con = sqlite3.connect(filename)
    cur = con.cursor()
    print(sqlite3_schema(database))
    cur.executescript(sqlite3_schema(database))

    for db_class in database.get_all_classes():
        attributes = [x.get_name() for x in db_class.get_all_components() if isinstance(x, Attribute)]
        columns = ', '.join(attributes)
        columns = ', '.join((columns, '_idx'))
        placeholders = ', '.join('?' for _ in range(len(attributes) + 1))
        sql = f"INSERT INTO {db_class.get_name()}({columns}) VALUES ({placeholders})"
        for idx, inst in enumerate(db_class.get_all_instances()):
            attrs_values = [getattr(inst, x) for x in attributes]
            attrs_values.append(idx)
            cur.execute(sql, attrs_values);

    con.commit()
    con.close()

def sqlite3_load(database, filename):
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

                if component.is_reference():
                    raise NotImplementedError

                if dtype.kind == 'u':
                    data = [int.from_bytes(x, 'little', signed=False) for (x,) in data]
                elif dtype.kind == 'i':
                    data = [int.from_bytes(x, 'little', signed=True) for (x,) in data]
                elif dtype == np.float32:
                    data = [struct.unpack('f', x) for (x,) in data]

                data = np.array(data, dtype=component.get_dtype())
                component.get_data()[idx] = data.flatten()

    con.close()

