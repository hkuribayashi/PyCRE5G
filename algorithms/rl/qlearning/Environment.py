import copy
import numpy as np

from algorithms.rl.qlearning.ActionSpace import ActionSpace
from algorithms.rl.qlearning.ObservationSpace import ObservationSpace


class Environment:

    def __init__(self, slice_):
        self.slice = slice_
        self.working_slice = copy.deepcopy(slice_)
        self.action_space = ActionSpace(len(self.working_slice.selected_bs) * 7)
        self.observation_space = ObservationSpace(101)
        self.priority_ues_weight = self.working_slice.cluster.bs_list[-1].hetnet.env.priority_ues_weight
        self.ordinary_ues_weight = self.working_slice.cluster.bs_list[-1].hetnet.env.ordinary_ues_weight

    def reset(self):
        self.working_slice = copy.deepcopy(self.slice)
        state = int(self.working_slice.cluster.evaluation['satisfaction'])
        return state

    def compute_satisfaction(self):
        fulfilled_qos_ues = np.array([ue for ue in self.working_slice.cluster.ue_list if ue.evaluation is True])
        total_priority_ues = len([ue for ue in self.working_slice.cluster.ue_list if ue.priority is True])
        total_ordinary_ues = len([ue for ue in self.working_slice.cluster.ue_list if ue.priority is False])

        weighted_sum = 0
        for ue in fulfilled_qos_ues:
            if ue.priority:
                weighted_sum += self.priority_ues_weight
            else:
                weighted_sum += self.ordinary_ues_weight
        total_weights = total_priority_ues * self.priority_ues_weight + total_ordinary_ues * self.ordinary_ues_weight

        return (weighted_sum / total_weights) * 100

    def step(self, action, state):

        total_actions = action + 1
        int_part = total_actions // 7
        remainder = total_actions % 7

        if int_part > 0:
            bs_index = int_part - 1
        else:
            bs_index = 0

        if remainder == 0:
            new_action = 6
        else:
            new_action = remainder - 1

        target_bs = self.working_slice.selected_bs[bs_index]

        if new_action == 0:
            target_bs.increase_bias(30.0)
        elif new_action == 1:
            target_bs.increase_bias(20.0)
        elif new_action == 2:
            target_bs.increase_bias(10.0)
        elif new_action == 3:
            target_bs.maintain_bias()
        elif new_action == 4:
            target_bs.increase_bias(-5.0)
        elif new_action == 5:
            target_bs.decrease_bias(-10.0)
        elif new_action == 6:
            target_bs.decrease_bias(-15.0)

        target_bs.hetnet.run(first_run_flag=False)
        info = self.compute_satisfaction()
        new_state = int(info)

        if new_state >= 95:
            reward = 10.0
            has_done = True
        elif new_state > state:
            reward = 0.01
            has_done = False
        elif new_state < state:
            reward = -0.01
            has_done = False
        else:
            reward = 0.0
            has_done = False

        return new_state, reward, has_done, info
