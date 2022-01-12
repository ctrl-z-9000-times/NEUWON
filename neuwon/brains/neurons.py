from collections import Iterable, Callable, Mapping
from neuwon.database import epsilon
from .growth import PathLengthCache, growth_algorithm
from . import regions
import numpy as np

class _Distribution:
    def __init__(self, arg):
        if isinstance(arg, Iterable):
            mean, std_dev = arg
            self.mean    = float(mean)
            self.std_dev = float(std_dev)
        else:
            self.mean    = float(arg)
            self.std_dev = 0.0
    def __call__(self, size=1):
        return np.random.normal(self.mean, self.std_dev, size=size)

class NeuronGrowthProgram:
    def __init__(self, brains, neuron_type, program):
        self.brains = brains
        self.neuron_type = str(neuron_type)
        self.neurons = []
        self.segments = []
        self.path_length_cache = PathLengthCache()
        self._run_program(*program)

    def _run_program(self, soma_parameters, *instructions_list):
        self._grow_soma(**soma_parameters)
        for parameters in instructions_list:
            self._run_growth_algorithm(**parameters)

    def _grow_soma(self, *,
                segment_type,
                region,
                diameter,
                number=None,
                density=None,
                mechanisms={},):
        region = self.brains.regions.make_region(region)
        assert (number is None) != (density is None), "'number' and 'density' are mutually exclusive."
        if number is not None:
            coordinates = [region.sample_point() for _ in range(number)]
        else:
            coordinates = region.sample_points(density)
        diameter = _Distribution(diameter)
        new_segments = []
        for c in coordinates:
            d = diameter()
            while d <= epsilon:
                d = diameter()
            n = self.brains.rxd_model.Neuron(c, d)
            self.neurons.append(n)
            new_segments.append(n.root)
        self.segments.extend(new_segments)
        self._insert_mechanisms(new_segments, mechanisms)

    def _run_growth_algorithm(self, *,
                segment_type,
                region,
                diameter,
                grow_from=None,
                exclude_from=None,
                morphology={},
                mechanisms={},):
        # Clean the inputs.
        segment_type = str(segment_type)
        region = self.brains.regions.make_region(region)

        if grow_from is None:
            roots = self.segments
        else:
            # grow_from is a segment type (or a list of them?)
            # It's a reference to a segment-type made in this program
            #       Raise an error if its not present / not grown yet.
            #       Only applies to the neurons made by *this* NeuronGrowthProgram.
            1/0 # TODO!

        neuron_region = morphology.pop('neuron_region', None)
        if neuron_region is not None:
            neuron_region = self.brains.regions.make_region(neuron_region)

        if exclude_from:
            1/0 # TODO!
            # exclude_from is a segment type (or list of them)
            #   The current growth step will not grow in the same neuron_region as the
            #   steps which produce those segment types.

        competitive = bool(morphology.pop('competitive', True))

        # Run the growth algorithm.
        if competitive:
            segments = growth_algorithm(roots, region,
                    path_length_cache=self.path_length_cache,
                    segment_parameters={
                            'segment_type': segment_type,
                            'diameter':     float(diameter),},
                    neuron_region=neuron_region,
                    **morphology)
        else:
            1/0 # TODO

        if neuron_region is not None:
            self.neuron_regions = regions.Union(self.neuron_regions, neuron_region)
        self.segments.extend(segments)
        self._insert_mechanisms(segments, mechanisms)

    def _insert_mechanisms(self, segments, mechanisms):
        # Clean the inputs.
        if isinstance(mechanisms, Mapping):
            pass
        elif isinstance(mechanisms, str):
            mechanisms = {mechanisms: {}}
        elif isinstance(mechanisms, Iterable):
            mechanisms = {mech_name: {} for mech_name in mechanisms}
        else: raise ValueError(f'Expected dictionary, not "{type(mechanisms)}"')
        # Lookup and then create the mechanism instances.
        for mech_name, parameters in mechanisms.items():
            mechanism = self.brains.rxd_model.mechanisms[mech_name]
            for segment in segments:
                mechanism(segment, **parameters)

class NeuronTypeFactory(dict):
    def __init__(self, brains, parameters: dict):
        super().__init__()
        self.brains = brains
        self.add_parameters(parameters)

    def add_parameters(self, parameters: dict):
        for neuron_type, program in parameters.items():
            self.add_neuron_type(neuron_type, program)

    def add_neuron_type(self, neuron_type: str, program: list):
        neuron_type = str(neuron_type)
        if neuron_type not in self:
            self[neuron_type] = NeuronGrowthProgram(self.brains, neuron_type, program).neurons
        return self[neuron_type]
