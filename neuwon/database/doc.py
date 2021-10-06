import textwrap

class Documentation:
    def __init__(self, name:str, doc:str=""):
        self.name = str(name)
        self.doc = textwrap.dedent(str(doc)).strip()

    def get_name(self) -> str: return self.name
    def get_doc(self) -> str:  return self.doc

    _name_doc = """
        Argument name
        """

    _doc_doc = """
        Argument doc is an optional documentation string.
        """
