from algorithms.si import PSOParticle


class StochasticIWPSOParticle(PSOParticle):

    def __init__(self, clustering_method, data_size, cognitive_factor):
        super().__init__(clustering_method, data_size, cognitive_factor)
