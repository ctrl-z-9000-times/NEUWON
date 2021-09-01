from neuwon.database import Database
import numpy as np
import numba

class GameOfLife:

    class _CellBaseClass:
        __slots__ = ()
        @classmethod
        def _add_to_database(cls, database):
            cell_data = database.add_class("Cell", cls)
            cell_data.add_attribute("coordinates", shape=(2,), dtype=np.int32)
            cell_data.add_attribute("alive", False, dtype=np.bool)
            cell_data.add_connectivity_matrix("neighbors", "Cell")
            return cell_data.get_instance_type()

    def __init__(self, shape):
        self.db = Database()
        self.Cell = self._CellBaseClass._add_to_database(self.db)
        self.shape = shape
        self.grid = np.empty(self.shape, dtype=object)
        for x in range(self.shape[0]):
            for y in range(self.shape[1]):
                self.grid[x,y] = self.Cell(coordinates=(x,y))
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

    def randomize(self, alive_fraction):
        a = self.db.get_data("Cell.alive")
        a.fill(False)
        a[np.random.uniform(size=a.shape) < alive_fraction] = True

    def get_num_alive(self):
        return sum(self.db.get_data("Cell.alive"))

    def advance(self):
        a = self.db.get_data("Cell.alive")
        n = self.db.get_data("Cell.neighbors")
        # C is the number of living neighbors for each cell.
        c = n * np.array(a, dtype=np.int32)
        _advance(a, c)

@numba.njit(parallel=True)
def _advance(a, c):
    for idx in numba.prange(len(a)):
        ci = c[idx]
        if a[idx]:
            if ci not in range(2, 4):
                a[idx] = False
        else:
            if ci == 3:
                a[idx] = True
