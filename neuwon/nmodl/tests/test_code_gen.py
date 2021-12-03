from neuwon.nmodl.code_gen import py_exec

def test_py_exec():
    py_exec("X = 3", globals())
    assert X == 3
    
    try: py_exec("1/0", globals())
    except ZeroDivisionError: pass
    else: 1/0
