

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
        for c in coordinates:
            d = diameter()
            while d <= epsilon:
                d = diameter()
            n = self.brains.rxd_model.Neuron(c, d)
            self.neurons.append(n)
            self.segments.append(n.root)

    def _run_growth_algorithm(self, *,
                segment_type,
                region,
                diameter,
                morphology={},
                mechanisms={},):

        region = self.brains.regions.make_region(region)

        relative_region = morphology.pop('relative_region', None)
        if relative_region is not None:
            relative_region = self.brains.regions.make_region(relative_region)

        grow_from_soma = bool(morphology.pop('grow_from_soma', False))
        if grow_from_soma:
            roots = [n.root for n in self.neurons]
        else:
            roots = self.segments

        competitive = bool(morphology.pop('competitive', True))
        if not competitive: 1/0 # TODO

        segments = growth_algorithm(roots, region,
                path_length_cache=self.path_length_cache,
                segment_parameters={
                        'segment_type': str(segment_type),
                        'diameter':     float(diameter),},
                **morphology)
        self.segments.extend(segments)
        # Insert the mechanisms.
        for mech_name, parameters in mechanisms.items():
            mechanism = 1/0
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
