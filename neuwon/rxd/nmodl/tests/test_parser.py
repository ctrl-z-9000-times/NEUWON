from neuwon.rxd.nmodl.parser import ANT, NmodlParser
import os.path

dirname = os.path.dirname(__file__)

def verify_file_parses(filename, check_for_v=True):
    x = NmodlParser(dirname + "/mod/" + filename)
    assert x.gather_documentation()
    assert x.gather_states()
    assert x.gather_parameters()
    b = x.gather_code_blocks()
    for z in b.values():
        z.gather_arguments()
    if check_for_v:
        assert 'v' in b['BREAKPOINT'].arguments

def test_hh():
    verify_file_parses("hh.mod")

def test_destexhe():
    verify_file_parses("Destexhe/gabaa5.mod")
    verify_file_parses("Destexhe/ampa5.mod", check_for_v=False)
    verify_file_parses("Destexhe/release.mod", check_for_v=False)

def test_nav11():
    verify_file_parses("Nav11.mod")
