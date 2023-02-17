import random
import numpy as np

from mobility.point import Point
from network.ue import UE
from utils.misc import get_ippp


class SimpleQueue:
    def __init__(self, env):
        self.arrival_rate = env.ue_arrival_rate
        self.service_rate = env.priority_ue_proportion
        self.total_time = env.total_time_steps
        self.simulation_area = env.simulation_area
        self.ue_height = env.ue_height
        self.priority_ue_proportion = env.priority_ue_proportion
        self.ue_height = env.ue_height
        self.total_priority_ues = 0
        self.total_ordinary_ues = 0
        self.max_associated_bs = env.max_bs_per_ue

    def populate_ues(self, user_density):
        ues = []
        x_UE, y_UE = get_ippp(self.simulation_area, user_density)
        total_ues = len(x_UE)
        self.total_priority_ues = int(total_ues * self.priority_ue_proportion)
        self.total_ordinary_ues = total_ues - self.total_priority_ues
        current_total_priority_ues = 0
        current_total_ordinary_ues = 0
        for idx in range(total_ues):
            p = Point(x_UE[idx], y_UE[idx], self.ue_height)
            ue = UE(idx, p)
            ue.max_associated_bs = self.max_associated_bs
            flag = False
            if not flag:
                value = random.randint(0, 1)
            else:
                value = 0
            if value is 1:
                if current_total_priority_ues < self.total_priority_ues:
                    ue.priority = True
                    current_total_priority_ues += 1
                else:
                    current_total_ordinary_ues += 1
            else:
                if flag or current_total_ordinary_ues < self.total_ordinary_ues:
                    current_total_ordinary_ues += 1
                else:
                    ue.priority = True
                    current_total_priority_ues += 1
            ues.append(ue)
        return ues

    def __str__(self):
        return '[priority_ues, ordinary_ues={}]'.format(self.total_priority_ues, self.total_ordinary_ues)
