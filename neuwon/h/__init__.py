from model import Model

_m = Model()

class Section:
    def __init__(self):
        1/0
        if self.name is None:
            self.name = "section_%d"%auto_inc

    def __str__(self):
        1/0

    # There are a lot of methods here...



_db = Database()

Cell = _db.add_class("Cell")
Section = _db.add_class("Section")

Section.add_attribute("name", None, dtype=object)
Section.add_attribute("cell", dtype=Cell)
