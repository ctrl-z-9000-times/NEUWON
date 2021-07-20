from .experiment import Experiment

def test_experiment():
    x = Experiment(20)
    x.run(100)
    x.model.db.check()
    x.analyze_grid_properties()
    x.find_alignment_points()
    x.select_exemplar_cells(.20)
