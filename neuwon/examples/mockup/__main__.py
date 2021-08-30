from neuwon.database import Database
from neuwon.opengl import Viewport
from neuwon.regions import Cylinder
from neuwon.segment import SegmentMethods
from neuwon.examples.mockup.model import Excitatory

diameter=100
height=20
region = Cylinder([0,0,0], [0,height,0], diameter/2)

db = Database()
SegmentMethods._initialize(db)
Segment = db.get("Segment").get_instance_type()
my_cell_type = Excitatory(Segment, region)
my_cell_type.grow(3)

view = Viewport()
view.set_scene(db)
while True:
    view.tick()
