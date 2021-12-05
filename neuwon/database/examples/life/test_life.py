from neuwon.database.examples.life.model import GameOfLife
import pytest

def test_smoke():
    model = GameOfLife((10, 10))
    model.randomize(.33)
    for step in range(100):
        model.advance()

def test_by_num_alive():
    size = 60
    model = GameOfLife((size, size))
    model.randomize(.33)
    num_cells = size * size
    sparsity = lambda: model.get_num_alive() / num_cells

    assert (.25 < sparsity() < .4)
    for _ in range(3): model.advance()
    assert (.20 < sparsity() < .33)
    for _ in range(500): model.advance()
    assert (.01 < sparsity() < .10)
    for _ in range(2000): model.advance()
    assert (.01 < sparsity() < .06)

    model.db.check()

@pytest.mark.skip()
def test_pickle():
    import pickle
    model = GameOfLife((10, 10))
    model.randomize(.33)
    model.advance()
    q = pickle.dumps(model)
    model2 = pickle.loads(q)
    model.advance()
    1/0
