from .model import GameOfLife
import argparse
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser("""Conway's Game of Life.""")
parser.add_argument('--size', type=int, default = 200,)
parser.add_argument('--sparsity', type=float, default = .1,)
parser.add_argument('--steps', type=int, default = 1000,)
parser.add_argument('--fps', type=int, default = 24,)
parser.add_argument('-o', '--out', type=str, default = "movie.gif",)
args = parser.parse_args()

model = GameOfLife((args.size, args.size))
model.randomize(args.sparsity)
t = 0

fig = plt.figure(figsize=(8, 8), dpi=180, constrained_layout=True)
ax = fig.subplots()
imgs = []
def save_frame():
    im = ax.imshow(1 - model.get_bitmap(), cmap='gray')
    im.axes.get_xaxis().set_visible(False)
    im.axes.get_yaxis().set_visible(False)
    imgs.append([im])

for i in range(args.fps // 2): save_frame()

for _ in range(args.steps):
    model.advance(); t += 1
    save_frame()

ani = animation.ArtistAnimation(fig, imgs, interval=1000/args.fps, blit=True)
ani.save(args.out)
