import random
import numpy as np

from config.network import Network
from mobility.point import Point
from network.bs import BS
from network.hetnet import HetNet

# Cria uma nova HetNet
h = HetNet(Network.DEFAULT)

# Gerando Macro Base Stations (MBSs)
ponto_mbs = Point(0.0, 0.0, Network.DEFAULT.mbs_height)
bs1 = BS(1, "MBS", ponto_mbs)

# Adiciona a MBS na Hetnet
h.add_bs(bs1)

# Tamanho "lado" do Cenário de Simulação
lado = np.sqrt(Network.DEFAULT.simulation_area)

# Gerando 10 novas SBS
for id_ in range(2, 12):
    x = random.uniform(-lado, lado)
    y = random.uniform(-lado, lado)
    p_x = Point(x, y, Network.DEFAULT.sbs_height)
    bs_x = BS(id_, "SBS", p_x)
    h.add_bs(bs_x)

# Roda a Hetnet apenas 1 Vez
for step in range(10):

    # Roda a HetNet com densidade aproximada de 200 UEs/km2
    h.run(user_density=200)

    # Imprime qual o grau de satisfção dos UEs
    print("Step: {} - Satisfação Global: {} | UEs: {}".format(step, h.evaluation['satisfaction'], len(h.ue_list)))

    # Se a safisfação dos UEs for menos que um valor defindo em Network.DEFAULT.outage_threshold
    if h.evaluation['satisfaction'] <= (Network.DEFAULT.outage_threshold * 100):

        # Apresenta uma represenação visual da Rede
        h.debug()

        # Nesse ponto o Algoritmo de AR deve ser chamado para aumentar a satisfação dos UES
