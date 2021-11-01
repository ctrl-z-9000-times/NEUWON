""" Implements the game logic for Conways Game of Life. """
from neuwon.database import Database, Compute
import numpy as np
import numba

class Cell:
    """ This class represents a square on the game board. """
    __slots__ = ()
    @classmethod
    def initialize(cls, database):
        cell_data = database.add_class(cls)
        cell_data.add_attribute("coordinates", shape=(2,), dtype=np.int32)
        cell_data.add_attribute("alive", False, dtype=np.int8)
        cell_data.add_connectivity_matrix("neighbors", "Cell")
        cell_data.add_attribute("count", dtype=np.int8,
                    doc="The number of living neighbors for this cell.")
        return cell_data.get_instance_type()

    @classmethod
    def advance(cls):
        cell_data = cls.get_database_class()
        a = cell_data.get_data("alive")
        n = cell_data.get_data("neighbors")
        c = cell_data.get_data("count")
        c[:] = n * a
        cls.advance_kernel() # Calls the kernel on all instances of the cell class.

    @Compute
    def advance_kernel(self):
        if self.alive:
            if self.count not in range(2, 4):
                self.alive = False
        else:
            if self.count == 3:
                self.alive = True


class GameOfLife:
    def __init__(self, shape):
        self.db = Database()
        self.Cell = Cell.initialize(self.db)
        self.shape = shape
        # Make all of the game Cells.
        self.grid = np.empty(self.shape, dtype=object)
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                self.grid[x,y] = self.Cell(coordinates=(x,y))
        # Setup the neighbors sparse-matrix.
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                cell = self.grid[x,y]
                neighbors = []
                for x_offset in [-1, 0, 1]:
                    for y_offset in [-1, 0, 1]:
                        nx = x - x_offset
                        ny = y - y_offset
                        if nx < 0: nx = 0
                        if ny < 0: ny = 0
                        if nx >= self.shape[0]: nx = self.shape[0] - 1
                        if ny >= self.shape[1]: ny = self.shape[1] - 1
                        neighbor = self.grid[nx, ny]
                        if cell != neighbor:
                            neighbors.append(neighbor)
                cell.neighbors = neighbors
        self.db.get("Cell.neighbors").to_csr()

    def randomize(self, fraction_alive):
        a = self.db.get_data("Cell.alive")
        a.fill(False)
        a[np.random.uniform(size=a.shape) < fraction_alive] = True

    def get_num_alive(self):
        return sum(self.db.get_data("Cell.alive"))

    def advance(self):
        self.Cell.advance()

