
class STDP:
    """ Spike Timing Dependent Plasticity

    References:
        Introduction to spiking neural networks:
        Information processing, learning and applications.
        Ponulak, Kasinski. 2011
        https://pubmed.ncbi.nlm.nih.gov/22237491/

    Citation for the model:
        Gerstner W, Kistler W (2002) Mathematical formulations
        of Hebbian learning. Biol Cybern 87: 404–415.
    """
    @staticmethod
    def initialize(synapse_data, *
            decay = 0.0,
            presyn_weight = 0.0,
            postsyn_weight = 0.0,
            hebbian_weight = 1e-2,
            anti_weight = -1e-2,
            hebbian_tau = 5.0,
            anti_tau = 5.0,):
        synapse_data.add_attribute()

