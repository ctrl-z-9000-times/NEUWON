from neuwon.nmodl.parser import ANT, NmodlParser

def verify_executes(filename):
    x = NmodlParser(filename)
    assert x.gather_documentation()
    assert x.gather_states()
    assert x.gather_units()
    assert x.gather_parameters()

    for AST in x.lookup(ANT.INITIAL_BLOCK):
        x.parse_code_block(AST)
    for AST in x.lookup(ANT.BREAKPOINT_BLOCK):
        x.parse_code_block(AST)
    for AST in x.lookup(ANT.DERIVATIVE_BLOCK):
        x.parse_code_block(AST)

def test_hh():
    verify_executes("./nmodl_library/hh.mod")

def test_nav11():
    verify_executes("./nmodl_library/Balbi2017/Nav11_a.mod")
