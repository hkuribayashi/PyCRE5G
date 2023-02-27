import random
import numpy as np

from config.DQNConfig import DQNConfig
from config.network import Network
from mobility.point import Point

from modules.rlm.RLM import RLM
from modules.rlm.ReinforcementLearningMethod import ReinforcementLearningMethod
from network.bs import BS
from network.hetnet import HetNet
from network.slice import Slice

# Cria uma nova HetNet
h = HetNet(Network.DEFAULT)

# Gerando Macro Base Stations (MBSs)
ponto_mbs = Point(0.0, 0.0, Network.DEFAULT.mbs_height)
bs1 = BS(1, "MBS", ponto_mbs)

# Adiciona a MBS na Hetnet
h.add_bs(bs1)

# Tamanho "lado" do Cenário de Simulação
lado = np.sqrt(Network.DEFAULT.simulation_area)/2

# Gerando 10 novas SBS
for id_ in range(2, 12):
    x = random.uniform(-lado, lado)
    y = random.uniform(-lado, lado)
    p_x = Point(x, y, Network.DEFAULT.sbs_height)
    bs_x = BS(id_, "SBS", p_x)
    h.add_bs(bs_x)

# Roda a Hetnet apenas 1 Vez
for step in range(10):

    # Roda a HetNet com densidade aproximada de 300 UEs/km2
    h.run(user_density=400)

    # Imprime qual o grau de satisfção dos UEs
    print("Execução: {} - Satisfação Global: {} | UEs: {}".format(step, h.evaluation['satisfaction'], len(h.ue_list)))

    # Se a safisfação dos UEs for menor que um valor defindo em Network.DEFAULT.outage_threshold
    if h.evaluation['satisfaction'] <= (Network.DEFAULT.outage_threshold * 100):

        # Apresenta uma representação visual da Rede
        h.debug()

        # Cria uma fatia de rede
        # TODO: Modificar este trecho para criar a fatia com uma parte da hetnet
        slice_ = Slice(h)

        # Chama o módulo de AR
        rlm = RLM(1, slice_)



