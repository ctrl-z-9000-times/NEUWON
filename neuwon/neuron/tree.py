from neuwon.database import Compute, NULL

class Tree:
    """
    Segments are organized in a tree.
    """
    __slots__ = ()
    @staticmethod
    def _initialize(database):
        db_cls = database.get_class('Segment')
        db_cls.add_attribute("parent", dtype=db_cls, allow_invalid=True)
        db_cls.add_connectivity_matrix("children", db_cls)

    def __init__(self, parent):
        self.parent = parent
        # Add ourselves to the parent's children list.
        parent = self.parent
        if parent is not None:
            siblings = parent.children
            siblings.append(self)
            parent.children = siblings

    @Compute
    def is_root(self) -> bool:
        return self.parent == NULL
