from neuwon.rxd.nmodl import cache
import tempfile

class _TestMechanism:
    def __init__(self):
        self.data = None

def test_cache():
    file = tempfile.NamedTemporaryFile(suffix=".mod", delete=False)
    file.close()

    py_obj = _TestMechanism()
    assert not cache.try_loading(file.name, py_obj)
    assert py_obj.data is None

    py_obj.data = "Hello cache!"
    cache.save(file.name, py_obj)

    py_obj2 = _TestMechanism()
    assert cache.try_loading(file.name, py_obj2)
    assert py_obj.data == py_obj2.data
