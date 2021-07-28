from database import Database
import numpy as np

class GameOfLife:

    class Cell:
        @classmethod
        def _add_to_database(cls, database):
            cell_data = database.add_class("Cell", cls)
            cell_data.add_attribute("coordinates", shape=(2,), dtype=np.int32)
            cell_data.add_attribute("alive", False, dtype=np.bool)
            cell_data.add_list_attribute("neighbors", dtype="Cell")
            return cell_data.get_instance_type()

    def __init__(self, size=(100,100)):
        self.db = Database()
        self.Cell = Game.Cell._add_to_database(self.db)
        grid = np.empty(size, dtype=object)
        for x in range(size[0]):
            for y in range(size[1]):
                grid[x,y] = self.Cell(coordinates=(x,y))
        for x in range(size[0]):
            for y in range(size[1]):
                cell = grid[x,y]
                for x_offset in [-1, 0, 1]:
                    for y_offset in [-1, 0, 1]:
                        nx = x - x_offset
                        ny = y - y_offset
                        if nx < 0: nx = 0
                        if ny < 0: ny = 0
                        if nx >= size[0]: nx = size[0] - 1
                        if ny >= size[1]: ny = size[1] - 1
                        neighbor = grid[nx, ny]
                        cell != neighbor:
                            cell.neighbors.append(neighbor)
        self.db.get("Cell.neighbors").to_array()

    def advance(self):
        a = self.db.get_data("Cell.alive")
        n = self.db.get_data("Cell.neighbors")
        c = n * a.reshape((-1, 1))
        for idx in range(len(a)):
            if a:
                if c != 2 and c != 3:
                    a[idx] = False
            else:
                if c == 3:
                    a[idx] = True
