import math
from operator import attrgetter

import numpy as np

from mobility.ueQeue import SimpleQueue
from mobility.point import Point
from network.bs import BS
from network.ne import NetworkElement
from utils.charts import get_visual
from utils.misc import get_pathloss, get_efficiency


class HetNet:

    def __init__(self, env):
        self.ue_list = list()
        self.list_bs = list()
        self.network_element = list()
        self.env = env
        self.evaluation = dict(satisfaction=0.0, total_ue=0.0)
        self.ueQueue = SimpleQueue(env)

    def populate_bs(self, number_bs):
        p0 = Point(0.0, 0.0, 35.0)
        mbs = BS(0, 'MBS', p0)
        self.add_bs(mbs)

        if number_bs == 200:
            step = 80
        elif number_bs == 400:
            step = 40
        else:
            step = 20

        counter = 1
        for y in range(-400, 440, 80):
            for x in range(-400, 420, step):
                if y != 0 and x != 0:
                    sbs_point = Point(x, y, 10.0)
                    sbs = BS(counter, 'SBS', sbs_point)
                    self.add_bs(sbs)
                    counter += 1

    def add_bs(self, bs):
        if bs.type == 'MBS':
            bs.power = self.env.mbs_power
            bs.tx_gain = self.env.mbs_gain
            bs.max_load = self.env.max_ue_per_mbs
        else:
            bs.power = self.env.sbs_power
            bs.tx_gain = self.env.sbs_gain
            bs.max_load = self.env.max_ue_per_sbs
        bs.hetnet = self
        self.list_bs.append(bs)

    def __get_ne(self):
        self.network_element = list()
        for ue in self.ue_list:
            linha_network_element = list()
            for bs in self.list_bs:
                linha_network_element.append(NetworkElement(ue, bs))
            self.network_element.append(linha_network_element)

    def run(self, user_density=300, first_run_flag=True):
        if first_run_flag:
            # Add UEs
            self.ue_list = self.ueQueue.populate_ues(user_density)

        if len(self.ue_list) > 0 and len(self.list_bs) > 0:

            if first_run_flag:
                # Constructs the NetworkElement structure
                self.__get_ne()

                # Compute SINR
                self.__get_sinr()

            # Compute UE x BS Association
            self.__get_association()

            # Compute Radio Resource Allocation
            self.__get_resource_allocation()

            # Compute UE Data Rate
            self.__get_ue_datarate()

            # Compute Performance Evaluation
            # if first_run_flag:
            self.__get_metrics()

    def reset(self):
        self.network_element = list()
        self.evaluation = dict(satisfaction=0.0, total_ue=0.0)

    def __get_sinr(self):
        bw = self.env.bandwidth * (10 ** 6)
        sigma = (10.0 ** (-3.0)) * (10.0 ** (self.env.noise_power / 10.0))
        total_thermal_noise = bw * sigma

        for linha in self.network_element:
            for element in linha:
                element.sinr = element.bs.power - get_pathloss(element.bs.type, element.distance) + element.bs.tx_gain
                element.sinr = (10 ** (-3.0)) * (10 ** (element.sinr / 10.0))
                other_elements = [x for x in linha if x != element]
                interference = 0.0
                for o_element in other_elements:
                    o_element_i = o_element.bs.power - get_pathloss(o_element.bs.type,
                                                                    o_element.distance) + o_element.bs.tx_gain
                    interference += ((10 ** (-3.0)) * (10 ** (o_element_i / 10.0)))
                element.sinr = element.sinr / (interference + total_thermal_noise)
                element.sinr = 10.0 * np.log10(element.sinr)
                element.biased_sinr = element.sinr

    def __get_association(self):
        for linha in self.network_element:
            ne = sorted(linha, key=attrgetter('biased_sinr'), reverse=True)
            for ne_element in ne:
                associated_bs = 0
                if ne_element.bs.load < ne_element.bs.max_load:
                    ne_element.coverage_status = True
                    ne_element.bs.load += 1
                    associated_bs += 1
                    if associated_bs >= ne_element.ue.max_associated_bs:
                        break

    def __get_resource_allocation(self):
        for coluna in map(list, zip(*self.network_element)):
            output = [element for element in coluna if element.coverage_status is True]
            bs_load = len(output)

            if bs_load > 0:
                total_priority = len([x for x in output if x.ue.priority is True])
                total_ue = bs_load
                total_non_priority = total_ue - total_priority

                teste_rb_por_ue = math.floor(output[0].bs.resouce_blocks/(total_priority * 2 + total_non_priority * 1))

                output[0].bs.load = bs_load
                rbs_per_ue = math.floor(output[0].bs.resouce_blocks / bs_load)
                for element in output:
                    peso = 2 if element.ue.priority is True else 1
                    element.ue.resource_blocks = teste_rb_por_ue * peso

    def __get_resource_allocation_2(self):
        for coluna in map(list, zip(*self.network_element)):
            output = [element for element in coluna if element.coverage_status is True]
            bs_load = len(output)

            if bs_load > 0:
                total_priority = len([x for x in output if x.ue.priority is True])
                total_ue = bs_load
                total_non_priority = total_ue - total_priority

                teste_rb_por_ue = output[0].bs.resouce_blocks/(total_non_priority * 2 + total_non_priority * 1)

                output[0].bs.load = bs_load
                rbs_per_ue = math.floor(output[0].bs.resouce_blocks / bs_load)
                for element in output:
                    element.ue.resource_blocks = rbs_per_ue

    def __get_ue_datarate(self):
        bitrate = self.env.number_subcarriers * self.env.number_ofdm_symbols
        for linha in self.network_element:
            ne = [element for element in linha if element.coverage_status is True]
            if len(ne) > 0:
                # TODO: Implementar associação de UE para várias BSs
                sinr = ne[0].sinr
                efficiency = get_efficiency(sinr)
                rbs = ne[0].ue.resource_blocks
                bitrate_ue = (rbs * efficiency * bitrate) / self.env.subframe_duration
                bitrate_ue = (bitrate_ue * 1000.0) / 1000000.0
                ne[0].ue.datarate = bitrate_ue
            else:
                linha[0].ue.datarate = 0

    def __get_metrics(self):
        fulfilled_qos_ues = np.array([ue for ue in self.ue_list if ue.evaluation is True])
        weighted_sum = 0
        for ue in fulfilled_qos_ues:
            if ue.priority:
                weighted_sum += self.env.priority_ues_weight
            else:
                weighted_sum += self.env.ordinary_ues_weight
        total_weights = self.ueQueue.total_priority_ues * self.env.priority_ues_weight + self.ueQueue.total_ordinary_ues * self.env.ordinary_ues_weight
        self.evaluation['satisfaction'] = (weighted_sum / total_weights) * 100
        self.evaluation['total_ue'] = len(self.ue_list)
        self.evaluation['total_priority_ues'] = self.ueQueue.total_priority_ues
        self.evaluation['total_ordinary_ues'] = self.ueQueue.total_ordinary_ues

    def debug(self):
        get_visual(self)
