
def browse_docs(database):
    md = generate_markdown(database)
    from subprocess import run, PIPE
    grip = run(["grip", "--browser", "-"], input=bytes(md, encoding='utf8'))

def generate_markdown(database):
    """ Markdown format documentation. """
    self = database
    case_insensitive = lambda kv_pair: kv_pair[0].lower()
    components = sorted(self.components.items(), key=case_insensitive)
    archetypes = sorted(self.archetypes.items(), key=case_insensitive)
    s  = "## Table of Contents\n"
    for name, obj in archetypes:
        s += "* [Archetype: %s](%s)\n"%(name, obj._markdown_link())
    s += "* [Index](#index)\n"
    for ark_name, ark in archetypes:
        s += str(ark) + "\n"
        s += "Components:\n"
        for comp_name, comp in components:
            if not comp_name.startswith(ark_name): continue
            s += "* [%s](%s)\n"%(comp_name, comp._markdown_link())
        s += "\n"
        for comp_name, comp in components:
            if not comp_name.startswith(ark_name): continue
            s += str(comp) + "\n"
    s += "---\n"
    s += "## Index\n"
    for name, obj in sorted(archetypes + components):
        s += "* [%s](%s)\n"%(name, obj._markdown_link())
    return s

