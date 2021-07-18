
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

def Component__str__(self):
        s = "### %s"%_DocString.__str__(self)
        if hasattr(self, "value"):
            s += "Value: %g"%(self.value)
            if self.units is not None: s += " " + self.units
            s += "\n\n"
        elif self.units is not None: s += "Units: %s\n\n"%self.units
        ref = getattr(self, "reference", False)
        if ref: s += "Reference to archetype [%s](%s).\n\n"%(ref.name, ref._markdown_link())
        if hasattr(self, "dtype") and not ref:
            s += "Data type: %s\n\n"%(self._dtype_name(),)
        lower_bound, upper_bound = self.valid_range
        if lower_bound is not None and upper_bound is not None:
            s += ""
        elif lower_bound is not None:
            s += ""
        elif upper_bound is not None:
            s += ""
        if getattr(self, "initial_value", None) is not None and not ref:
            s += "Initial Value: %g"%(self.initial_value)
            if self.units is not None: s += " " + self.units
            s += "\n\n"
        if self.allow_invalid:
            if ref:
                s += "Value may be NULL.\n\n"
            else: s += "Value may be NaN.\n\n"
        return s

def classtype__str__(self):
        s = "---\n## %s"%_DocString.__str__(self)
        return s


class _DocString:
    def _class_name(self):
        return type(self).__name__.replace("_", " ").strip()

    def _markdown_header(self):
        return "%s: %s"%(self._class_name(), self.name)

    def _markdown_link(self):
        name = "#" + self._markdown_header()
        substitutions = (
            (":", ""),
            ("/", ""),
            (" ", "-"),
        )
        for x in substitutions: name = name.replace(*x)
        return name.lower()

    def __str__(self):
        anchor = "<a name=\"%s\"></a>"%self.name
        return "%s%s\n%s\n\n"%(self._markdown_header(), anchor, self.doc)


