from neuwon.database import *
import math
import random
from htm import SDR, Metrics

class NN:
    def __init__(self):
        self.init_database()
        self.init_model()

    def init_database(self):
        self.db = Database()
        self.Cell = self.db.add_class("Cell")
        self.Cell.add_attribute("activity_level", initial_value=0)
        self.ExcitSyn = self.init_syn_class("ExcitSyn")
        self.InhibSyn = self.init_syn_class("InhibSyn")
        self.Cell.add_attribute("excit_level")
        self.Cell.add_attribute("inhib_level")

    def init_syn_class(self, name):
        syn_class = self.db.add_class(name)
        syn_class.add_attribute("weight")
        syn_class.add_attribute("presyn", dtype=self.Cell)
        syn_class.add_attribute("postsyn", dtype=self.Cell)
        return syn_class

    def init_model(self):
        self.num_excit = 200
        self.num_inhib =  10
        self.num_syn   =  40
        self.excit_weight = lambda: random.random() / self.num_syn
        self.inhib_weight = lambda: random.random() / 10
        self.excit_thresh = 0
        self.excit_slope  = 2
        self.inhib_thresh = 0
        self.inhib_slope  = 2

        self.excit_cells = [self.Cell() for _ in range(self.num_excit)]
        self.inhib_cells = [self.Cell() for _ in range(self.num_inhib)]
        self.all_cells   = self.excit_cells + self.inhib_cells
        self.excit_syn   = []
        self.inhib_syn   = []
        for cell in self.all_cells:
            for _ in range(self.num_syn):
                if random.random() < len(self.excit_cells) / len(self.all_cells):
                    self.excit_syn.append(self.ExcitSyn(
                            presyn=random.choice(self.excit_cells),
                            postsyn=cell,
                            weight=self.excit_weight()))
                else:
                    self.inhib_syn.append(self.InhibSyn(
                            presyn=random.choice(self.inhib_cells),
                            postsyn=cell,
                            weight=self.inhib_weight()))

    def activate_random_cells(self, num):
        for cell in random.sample(self.all_cells, num):
            cell.activity_level = 1

    def advance(self):
        self.Cell.get_component("excit_level").get().fill(0)
        self.Cell.get_component("inhib_level").get().fill(0)
        for syn in self.excit_syn:
            syn.postsyn.excit_level += syn.weight * syn.presyn.activity_level
        for syn in self.inhib_syn:
            syn.postsyn.inhib_level += syn.weight * syn.presyn.activity_level
        for cell in self.excit_cells:
            activation_function(cell, self.excit_thresh, self.excit_slope)
        for cell in self.inhib_cells:
            activation_function(cell, self.inhib_thresh, self.inhib_slope)

    def learn(self):
        for syn in self.excit_syn:
            pass

def activation_function(cell, thresh, gain):
    h = cell.excit_level / (1. + cell.inhib_level)
    h = h - thresh
    if h <= 0: y = 0
    else: y = 2 / math.pi * math.atan(gain * h)
    cell.activity_level = y

def test_nn():
    nn = NN()
    x = SDR(len(nn.all_cells))
    m = Metrics(x, 30)
    nn.activate_random_cells(50)
    for _ in range(100):
        x.sparse = np.nonzero(nn.Cell.get_component("activity_level").get() > .1)[0]
        print("Sparsity", x.getSparsity())
        nn.advance()
    print(m)

if __name__ == "__main__":
    test_nn()
