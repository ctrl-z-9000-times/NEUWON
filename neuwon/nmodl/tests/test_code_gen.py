from neuwon.nmodl.code_gen import mangle, demangle, mangle2, demangle2, py_exec

def test_mangle():
    test_strs = "a ABC a123 _123 _ __".split()
    for s in test_strs:
        assert s == demangle(mangle(s))
        assert s == demangle2(mangle2(s))
        assert s != mangle(s)
        assert s != mangle2(s)
        assert mangle(s) != mangle2(s)

def test_py_exec():
    py_exec("X = 3", globals())
    assert X == 3
    
    try: py_exec("1/0", globals())
    except ZeroDivisionError: pass
