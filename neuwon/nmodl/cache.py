""" Private module. """
__all__ = []

from os.path import abspath, join, exists, getmtime
from sys import stderr
from zlib import crc32
import os
import pickle

def _dir_and_file(filename):
    cache_dir = abspath(".nmodl_cache")
    filename  = abspath(filename)
    cache_file = join(cache_dir, "%X.pickle"%crc32(bytes(filename, 'utf8')))
    return (cache_dir, cache_file)

def try_loading(filename, obj):
    """ Returns True on success, False indicates that no changes were made to the object. """
    cache_dir, cache_file = _dir_and_file(filename)
    if not exists(cache_file): return False
    try:
        nmodl_ts  = getmtime(filename)
        cache_ts  = getmtime(cache_file)
    except FileNotFoundError: return False
    python_ts = getmtime(__file__)
    if nmodl_ts > cache_ts: return False
    if python_ts > cache_ts: return False
    try:
        with open(cache_file, 'rb') as f: data = pickle.load(f)
    except Exception as err:
        print("Warning: nmodl cache read failed:", str(err), file=stderr)
        return False
    obj.__dict__.update(data)
    return True

def save(filename, obj):
    cache_dir, cache_file = _dir_and_file(filename)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f: pickle.dump(obj.__dict__, f)
    except Exception as x:
        print("Warning: nmodl cache write failed:", str(x), file=stderr)
