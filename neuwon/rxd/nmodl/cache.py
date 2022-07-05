from os import makedirs, listdir
from os.path import abspath, dirname, join, exists, getmtime
from sys import stderr
from zlib import crc32
import pickle

def _dir_and_file(filename, parameters: dict):
    cache_dir  = abspath(".nmodl_cache")
    filename   = abspath(filename)
    parameters = ','.join(str(v) for k,v in sorted(parameters.items()))
    hash_src   = f'{filename},{parameters}'
    cache_file = join(cache_dir, "%X.pickle"%crc32(bytes(filename, 'utf8')))
    return (cache_dir, cache_file)

def try_loading(filename, parameters, obj):
    """ Returns True on success, False indicates that no changes were made to the object. """
    cache_dir, cache_file = _dir_and_file(filename, parameters)
    if not exists(cache_file): return False
    # Check file modification time stamps.
    try:
        nmodl_ts  = getmtime(filename)
        cache_ts  = getmtime(cache_file)
    except FileNotFoundError: return False
    if nmodl_ts > cache_ts: return False
    # Check that the nmodl module is older than the cache too.
    src_dir = dirname(__file__)
    for src_file in listdir(src_dir):
        if src_file.endswith('.py'):
            src_ts = getmtime(join(src_dir, src_file))
            if src_ts > cache_ts: return False
    # Load the cache.
    try:
        with open(cache_file, 'rb') as f:
            cache_obj = pickle.load(f)
    except Exception as err:
        print("Warning: nmodl cache read failed:", str(err), file=stderr)
        return False
    obj.__class__ = cache_obj.__class__
    obj.__dict__.update(cache_obj.__dict__)
    return True

def save(filename, parameters, obj):
    cache_dir, cache_file = _dir_and_file(filename, parameters)
    try:
        makedirs(cache_dir, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as x:
        print("Warning: nmodl cache write failed:", str(x), file=stderr)
