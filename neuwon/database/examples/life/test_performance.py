from neuwon.database.examples.life.model import GameOfLife
import argparse
import numpy as np

parser = argparse.ArgumentParser("""
Performance test with
    Conway's Game of Life

Run as fast as possible, with no graphical output.
""")
parser.add_argument('--steps', type=int, default = 1000*1000,)
parser.add_argument('--size', type=int, default = 1000,)
args = parser.parse_args()

model = GameOfLife((args.size, args.size))
model.randomize(.33)

for step in range(args.steps):
    model.advance()
