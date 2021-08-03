from neuwon.database.examples.KropffTreves2008.experiment import Experiment

def test():
    x = Experiment(20)
    x.run(100)
    x.model.db.check()
    x.analyze_grid_properties()
    x.find_alignment_points()
    x.select_exemplar_cells(.20)

def test_multiple_model():
    x = Experiment(20)
    y = Experiment(19)
    x.run(1)
    y.run(2)
    z = Experiment(21)
    y.run(3)
    z.run(10)
    x.model.db.check()
    y.model.db.check()
    z.model.db.check()
