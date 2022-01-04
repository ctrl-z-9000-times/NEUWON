
class NeuronTypesFactory(dict):
    def __init__(self, parameters: dict):
        super().__init__()
        self.add_parameters(parameters)

    def add_parameters(self, parameters: dict):
        self.parameters = parameters
        for name, parameters in self.parameters.items():
            self.add_neuron_type(name, *parameters)
        del self.parameters

    def add_neuron_type(self, name: str, args, kwargs) -> GrowthRoutine:
        if name in self:
            return self[name]
        self[name] = rgn = self.make_routine(routine_type, args, kwargs)
        return rgn

    # _routine_types = {cls.__name__: cls for cls in
    #         (Soma, Dendrite)}

    # def make_routine(self, routine_type, args, kwargs) -> GrowthRoutine:
        # if 
        # if isinstance(args, GrowthRoutine):
        #     return args
        # elif isinstance(args, str):
        #     region_name = args
        #     if region_name not in self:
        #         region_parameters = self.parameters[region_name]
        #         self.add_routine(region_name, region_parameters)
        #     return self[region_name]
        # elif isinstance(args, Iterable):
        #     region_type, *args = args
        #     if region_type in ('Intersection', 'Union', 'Not'):
        #         args = [self.make_routine(r) for r in args]
        #     return self._routine_types[region_type](*args)
        # else:
        #     raise ValueError(args)




################################################################################


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



class NeuronType:
    def __init__(self, brain, soma):
        self.Segment = Segment
        self.region = region
        self.diameter = _Distribution(diameter)
        self.segments = []
        assert(isinstance(self.region, Region))
        assert(self.diameter.mean - 2 * self.diameter.std_dev > 0)

    def _init_soma(self, diameter, region):


    def grow(self, num_cells):
        new_segments = []
        for _ in range(num_cells):
            coordinates = self.region.sample_point()
            diameter = self.diameter()
            while diameter <= epsilon:
                diameter = self.diameter()
            x = self.Segment(None, coordinates, diameter)
            new_segments.append(x)
        self.segments.extend(new_segments)
        return new_segments


