from neuwon.nmodl.parser import ANT, NmodlParser

def verify_file_parses(filename):
    x = NmodlParser(filename)
    assert x.gather_documentation()
    assert x.gather_states()
    assert x.gather_units()
    assert x.gather_parameters()
    b = x.gather_code_blocks()
    for z in b.values():
        z.gather_arguments()
    assert 'v' in b['BREAKPOINT'].arguments

def test_hh():
    verify_file_parses("./nmodl_library/hh.mod")

def test_nav11():
    verify_file_parses("./nmodl_library/Balbi2017/Nav11_a.mod")
