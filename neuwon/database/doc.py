import textwrap

class Documentation:
    """
    This class provides two public attributes, both are strings:
      * name
      * doc
    """
    def __init__(self, name:str, doc:str=''):
        self.name = str(name)
        assert '.' not in self.name
        if not doc: doc = ''
        self.doc = textwrap.dedent(str(doc)).strip()

    def get_name(self) -> str:  return self.name
    def get_doc(self) -> str:   return self.doc
