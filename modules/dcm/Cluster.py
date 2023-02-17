import numpy as np


class Cluster:

    def __init__(self, id_, priority_ues_weight, ordinary_ues_weight, outage_threshold):
        self.id = id_
        self.ue_list = []
        self.bs_list = None
        self.evaluation = {}
        self.target_cluster = False
        self.priority_ues_weight = priority_ues_weight
        self.ordinary_ues_weight = ordinary_ues_weight
        self.outage_threshold = outage_threshold
        self.network_slice = None

    def evaluate(self, flag=True):
        fulfilled_qos_ues = np.array([ue for ue in self.ue_list if ue.evaluation is True])
        weighted_sum = 0
        for ue in fulfilled_qos_ues:
            if ue.priority:
                weighted_sum += self.priority_ues_weight
            else:
                weighted_sum += self.ordinary_ues_weight
        total_ues = len(self.ue_list)
        total_priority_ues = len([ue.priority for ue in self.ue_list if ue.priority is True])
        total_weights = total_priority_ues * self.priority_ues_weight + (
                    total_ues - total_priority_ues) * self.ordinary_ues_weight

        evaluation = weighted_sum / total_weights
        if flag and evaluation < self.outage_threshold:
            self.target_cluster = True
        else:
            self.target_cluster = True

        self.evaluation['satisfaction'] = evaluation * 100
        self.evaluation['total_ue'] = total_ues
        self.evaluation['total_priority_ues'] = total_priority_ues
        self.evaluation['total_ordinary_ues'] = total_ues - total_priority_ues

    def __str__(self):
        return "[id={}, evaluation={}, target_cluster={}, total_ues={}, " \
               "total_priority_ues={}, total_ordinary_ues={}]".format(self.id,
                                                                      self.evaluation['satisfaction'],
                                                                      self.target_cluster,
                                                                      self.evaluation['total_ue'],
                                                                      self.evaluation["total_priority_ues"],
                                                                      self.evaluation["total_ordinary_ues"])
