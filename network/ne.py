from utils.misc import get_distance


class NetworkElement:

    def __init__(self, ue, bs):
        self.ue = ue
        self.bs = bs
        self.sinr = 0.0
        self.biased_sinr = 0.0
        self.distance = get_distance(ue.point, bs.point)
        self.coverage_status = False

    def __str__(self):
        return 'Network Element: ue={}, bs={}, sinr={}, biased_sinr={}, distance={}, coverage_status={}'.format(self.ue, self.bs,
                                                                                                                self.sinr,
                                                                                                                self.biased_sinr,
                                                                                                                self.distance,
                                                                                                                self.coverage_status)
