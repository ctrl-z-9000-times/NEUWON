
class AP_Queue:
    # A mix-in for synapses?
    @staticmethod
    def init_database(synapse_data: 'DB_Class'):
        database = synapse_data.get_database()
        event_data = database.add_class(synapse_data.get_name() + '_AP_Event',
                                        sort_key=('delivery_time', 'synapse'))
        event_data.add_attribute('synapse', dtype=synapse_data)
        event_data.add_attribute('delivery_tick', dtype=int)

        neuron_data = database.get_class('Neuron')
        synapse_cls = synapse_data.get_instance_type()
        synapse_cls.matrix_name = '_' + synapse_data.get_name()
        synapse_cls.matrix = neuron_data.add_sparse_matrix(
                                        synapse_data.matrix_name, synapse_data)

    def init_instance(self, presynapse: 'Neuron', propagation_delay=0.0):
        matrix = getattr(presynapse, self.matrix_name)
        matrix.append((presynapse, propagation_delay))
        setattr(presynapse, self.matrix_name, matrix)

    @classmethod
    def dispatch(cls, presynaptic_APs: [int]):
        targets = cls.matrix.get_rows(presynaptic_APs)
        # How to do the manipulations to go from AP's through the matrix and
        # into the events structures, and to do it efficiently?
        #       Start by doing it inefficiently and make unit tests.
        #       Then optimize as needed.
        1/0

    @classmethod
    def collect(cls, time):
        """ Returns an array of the synapse indexes which just received an AP. """
        assert cls.get_database.is_sorted()
        # uhhg i dont want to implement a bisect sort for this...
