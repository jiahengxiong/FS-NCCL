import copy
import itertools

import networkx as nx
import matplotlib.pyplot as plt
from utils import Network
from collections import defaultdict
from SIM.simulator import simulate_allgather_event_driven
import pickle

collectives = 100100
for collective in range(collectives):
    network = Network.Network()
    network.build_link_intraDC()
    G = network.topology
    path = "TOPO/" + str(collective) + ".pkl"

    with open(path, "wb") as f:
        pickle.dump(G, f)