from neuwon.database.examples.life.model import GameOfLife
import argparse

parser = argparse.ArgumentParser("""
Performance test with Conway's Game of Life.

To run as fast as possible: there is no graphical output.
""")
parser.add_argument('--size', type=int, default = 500,)
parser.add_argument('--steps', type=int, default = 10*1000,)
parser.add_argument('--target', choices=('host', 'cuda'), default='host')
args = parser.parse_args()

model = GameOfLife((args.size, args.size))
model.randomize(.33)

with model.db.using_memory_space(args.target):
    for step in range(args.steps):
        model.advance()
        if step % 1000 == 0: print(f'Completed {step+1} steps.')
