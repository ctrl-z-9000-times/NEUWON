import argparse
from .experiment import Experiment

parser = argparse.ArgumentParser("""
This program reproduces the figures and diagrams in:
    The emergence of grid cells: intelligent design or just adaptation? 
    Emilio Kropff and Alessandro Treves, 2008
""")
parser.add_argument('--steps', type=int, default = 1000 * 1000,)
parser.add_argument('--size', type=int, default = 200,)
args = parser.parse_args()

x = Experiment(args.size)
x.run(args.steps)
x.analyze_grid_properties()
x.find_alignment_points()
x.select_exemplar_cells(20)
x.plot()
