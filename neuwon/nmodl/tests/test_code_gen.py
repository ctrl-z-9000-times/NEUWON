from neuwon.nmodl.code_gen import mangle, demangle, mangle2, demangle2, py_exec

def test_mangle():
    test_strs = "a ABC a123 _123 456_".split()
    for s in test_strs:
        assert s == demangle(mangle(s))
        assert s == demangle2(mangle2(s))
        assert mangle(s) != mangle2(s)
        assert mangle(s) not in test_strs
        assert mangle2(s) not in test_strs

def test_py_exec():
    py_exec("X = 3", globals())
    assert X == 3
    
    try: py_exec("1/0", globals())
    except ZeroDivisionError: pass
