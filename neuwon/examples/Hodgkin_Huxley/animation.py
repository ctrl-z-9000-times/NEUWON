""" Model of an action potential propagating through an axonal arbor. """

from neuwon.database.time import TimeSeries
from neuwon.growth import *
from neuwon.model import *
from neuwon.gui.viewport import *
from neuwon.regions import *
import numpy as np

min_v = -90.
max_v = +70.

class main:
    def __init__(self,
            time_step = .1,):
        self.time_step      = time_step

        self.make_model()
        self.run_GUI()

    def make_region(self):
        return Sphere([0, 0, 0], 500)
        r = 30
        w = 100
        return Union([
            Sphere(  [-w, -w, 0], r),
            Cylinder([-w, -w, 0], [-w, w, 0], r),
            Sphere(  [-w, w, 0], r),
            Cylinder([-w, w, 0], [w, w, 0], r),
            Sphere(  [w, w, 0], r),
            Cylinder([w, w, 0], [w, -w, 0], r),
            Sphere(  [w, -w, 0], r),
            Cylinder([w, -w, 0], [0, -w, w*2], r),
            Sphere(  [0, -w, w*2], r),
            Cylinder([0, -w, w*2], [0, 0, w*6], r),
            Sphere(  [0, 0, w*6], r),
        ])

    def make_model(self):
        self.model = m = Model(self.time_step, celsius = 6.3)
        m.add_species("na", reversal_potential = +60)
        m.add_species("k",  reversal_potential = -88)
        m.add_species("l",  reversal_potential = -54.3,)
        HH = m.add_reaction("./nmodl_library/hh.mod")
        self.soma = m.Segment(None, [0,0,0], 10)
        self.axon = Tree(self.soma, self.make_region(), 0.000025,
            balancing_factor = .0,
            extension_angle = np.pi / 6,
            extension_distance = 60,
            bifurcation_angle = np.pi / 3,
            bifurcation_distance = 40,
            extend_before_bifurcate = True,
            maximum_segment_length = 30,
            diameter = .5)
        self.axon.grow()
        self.axon = self.axon.get_segments()
        HH(self.soma, scale=1)
        for seg in self.axon:
            HH(seg)
        if True:
            print("Number of Locations:", len(self.model))
            sa_units = self.soma.get_database_class().get("surface_area").get_units()
            sa = self.soma.surface_area
            print("Soma surface area:", sa, sa_units)
            sa += sum(x.surface_area for x in self.axon)
            print("Total surface area:", sa, sa_units)

    def run_GUI(self):
        view = Viewport()
        view.set_scene(self.model)
        voltage = self.model.Segment.get_database_class().get("voltage")

        period = 30
        cooldown = 0
        while True:
            if cooldown <= 0:
                self.soma.inject_current(2e-9, 1)
                cooldown = period / self.time_step
            else:
                cooldown -= 1
            v = ((voltage.get_data() - min_v) / (max_v - min_v)).clip(0, 1)
            colors = [(x, 0, 1-x) for x in v]
            view.tick(colors)
            self.model.advance()

if __name__ == "__main__": main()
