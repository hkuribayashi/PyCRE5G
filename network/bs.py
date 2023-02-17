class BS:

    def __init__(self, id_, type_, point):
        self.id = id_
        self.type = type_
        self.point = point
        self.load = 0.0
        self.max_load = 0.0
        self.power = 0.0
        self.tx_gain = 0.0
        self.resouce_blocks = 100
        self.hetnet = None

    def increase_bias(self, bias, ue_list=None):
        if ue_list is None:
            for coluna in map(list, zip(*self.hetnet.network_element)):
                if coluna[0].bs.id == self.id:
                    for ne in coluna:
                        if abs(ne.sinr - ne.biased_sinr) < self.hetnet.env.max_bias:
                            ne.biased_sinr += bias
        else:
            for coluna in map(list, zip(*self.hetnet.network_element)):
                if coluna[0].bs.id == self.id:
                    for ne in coluna:
                        if abs(ne.sinr - ne.biased_sinr) < self.hetnet.env.max_bias and any(x.id == ne.ue.id for x in ue_list):
                            ne.biased_sinr += bias

    def decrease_bias(self, bias):
        for coluna in map(list, zip(*self.hetnet.network_element)):
            if coluna[0].bs.id == self.id:
                for ne in coluna:
                    if abs(ne.sinr - ne.biased_sinr) < abs(self.hetnet.env.min_bias):
                        ne.biased_sinr += bias

    def maintain_bias(self):
        pass

    def __lt__(self, other):
        return self.id < other.id

    def __str__(self):
        return 'BS id={}, type={}, load={}'.format(self.id, self.type, self.load)
