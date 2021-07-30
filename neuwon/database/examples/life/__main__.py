from neuwon.database.examples.life.model import GameOfLife
import argparse
import numpy as np
import pygame
from pygame.locals import *
pygame.init()

parser = argparse.ArgumentParser("""Conway's Game of Life.""")
parser.add_argument('--size', type=int, default = 200,)
parser.add_argument('--sparsity', type=float, default = .33,)
parser.add_argument('--fps', type=int, default = 24,)
args = parser.parse_args()

model = GameOfLife((args.size, args.size))
model.randomize(args.sparsity)
t = 0

sqr_sz = 5
size = [sqr_sz*args.size] * 2
screen = pygame.display.set_mode(size, pygame.HWSURFACE | pygame.DOUBLEBUF)
BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
pygame.display.set_caption("Conway's Game of Life")
clock = pygame.time.Clock()

while True:
    pygame.display.update()
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            break

    if t > args.fps / 2:
        model.advance()
    t += 1

    screen.fill(WHITE)
    a = model.db.get_data("Cell.alive")
    c = model.db.get_data("Cell.coordinates")
    for idx in np.nonzero(a)[0]:
        x, y = c[idx]
        pygame.draw.rect(screen, BLACK, [x*sqr_sz, y*sqr_sz, sqr_sz, sqr_sz])

    pygame.display.flip()
    clock.tick(args.fps)
