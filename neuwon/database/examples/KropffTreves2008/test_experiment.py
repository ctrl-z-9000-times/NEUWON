from neuwon.database.examples.KropffTreves2008.experiment import Experiment

def test():
    x = Experiment(20)
    x.run(100)
    x.model.db.check()
    x.analyze_grid_properties()
    x.find_alignment_points()
    x.select_exemplar_cells(.20)

# TODO: Make two versions of the same database and verify that they operate
# independently.

# def test_multi_model():
#     pass
