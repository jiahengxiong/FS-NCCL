import networkx as nx
import uuid
from utils.Traffic import get_bandwidth_trace


def add_sym_rand_link(G, u, v, cap_base=100, delay=1e-5, **extra_attrs):
    """
    为 (u <-> v) 添加对称链路：
      - 对这一对节点 {u,v} 采样一次 capacity，并在两个方向复用
      - 不同的节点对会重新采样（因为是不同的函数调用）
    """
    cap = get_bandwidth_trace(method='random', steps=1, capacity=cap_base)
    attrs = dict(extra_attrs)
    attrs.update({
        "capacity": cap,
        "propagation_delay": delay
    })
    G.add_edge(u, v, **attrs)
    G.add_edge(v, u, **attrs)


class Network:
    def __init__(self):
        G = self.get_topology()
        self.topology = G
        self.relabel_non_gpu_nodes()

    def get_topology(self):
        G = nx.DiGraph()
        G.add_node(0, type='GPU', field='DC0', DC=0)
        G.add_node(1, type='GPU', field='DC0', DC=0)
        G.add_node(2, type='GPU', field='DC0', DC=0)
        G.add_node(3, type='GPU', field='DC0', DC=0)
        G.add_node(4, type='GPU', field='DC1', DC=1)
        G.add_node(5, type='GPU', field='DC1', DC=1)
        G.add_node(6, type='GPU', field='DC1', DC=1)
        G.add_node(7, type='GPU', field='DC1', DC=1)

        G.add_node('Border Switch 0', type='Boarder Switch', field='WAN', DC=0)
        G.add_node('Border Switch 1', type='Boarder Switch', field='WAN', DC=0)
        G.add_node('Border Switch 2', type='Boarder Switch', field='WAN', DC=1)
        G.add_node('Border Switch 3', type='Boarder Switch', field='WAN', DC=1)

        G.add_node('Router 0', type='Router', field='WAN')
        G.add_node('Router 1', type='Router', field='WAN')
        G.add_node('Router 2', type='Router', field='WAN')
        G.add_node('Router 3', type='Router', field='WAN')
        G.add_node('Router 4', type='Router', field='WAN')

        add_sym_rand_link(G, 0, 'Border Switch 0', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 0, 'Border Switch 1', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 1, 'Border Switch 0', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 1, 'Border Switch 1', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 2, 'Border Switch 0', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 2, 'Border Switch 1', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 3, 'Border Switch 0', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 3, 'Border Switch 1', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 'Border Switch 0', 'Router 0', cap_base=100, delay=1e-5)
        add_sym_rand_link(G, 'Border Switch 1', 'Router 0', cap_base=100, delay=1e-5)

        add_sym_rand_link(G, 'Router 0', 'Router 1', cap_base=100, delay=1e-5)
        add_sym_rand_link(G, 'Router 0', 'Router 4', cap_base=100, delay=1e-5)
        add_sym_rand_link(G, 'Router 1', 'Router 2', cap_base=100, delay=1e-5)
        add_sym_rand_link(G, 'Router 2', 'Router 3', cap_base=100, delay=1e-5)
        add_sym_rand_link(G, 'Router 2', 'Router 4', cap_base=100, delay=1e-5)
        add_sym_rand_link(G, 'Router 3', 'Router 4', cap_base=100, delay=1e-5)

        add_sym_rand_link(G, 'Border Switch 2', 'Router 3', cap_base=100, delay=1e-5)
        add_sym_rand_link(G, 'Border Switch 3', 'Router 3', cap_base=100, delay=1e-5)

        add_sym_rand_link(G, 4, 'Border Switch 2', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 4, 'Border Switch 3', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 5, 'Border Switch 2', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 5, 'Border Switch 3', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 6, 'Border Switch 2', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 6, 'Border Switch 3', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 7, 'Border Switch 2', cap_base=200, delay=7e-7)

        add_sym_rand_link(G, 7, 'Border Switch 3', cap_base=200, delay=7e-7)
        # self.build_link_intraDC()



        return G

    def build_link_intraDC(self):
        DC_0_GPU_list = []
        for node in self.topology.nodes():
            if self.topology.nodes[node]['type'] == 'GPU' and self.topology.nodes[node]['field'] == 'DC0':
                DC_0_GPU_list.append(node)
        for i in range(len(DC_0_GPU_list)):
            for j in range(i + 1, len(DC_0_GPU_list)):
                if i != j:
                    add_sym_rand_link(self.topology, DC_0_GPU_list[i], DC_0_GPU_list[j], cap_base=200, delay=7e-7)

        DC_1_GPU_list = []
        for node in self.topology.nodes():
            if self.topology.nodes[node]['type'] == 'GPU' and self.topology.nodes[node]['field'] == 'DC1':
                DC_1_GPU_list.append(node)
        for i in range(len(DC_1_GPU_list)):
            for j in range(i + 1, len(DC_1_GPU_list)):
                if i != j:
                    add_sym_rand_link(self.topology, DC_1_GPU_list[i], DC_1_GPU_list[j], cap_base=200, delay=7e-7)

    def relabel_non_gpu_nodes(self):
        """把非GPU节点改成从最大GPU编号+1开始的整数"""
        G = self.topology
        max_gpu_id = max(n for n in G.nodes if isinstance(n, int))  # GPU节点编号
        next_id = max_gpu_id + 1
        mapping = {}

        for node in list(G.nodes):
            if not isinstance(node, int):  # 非GPU节点
                mapping[node] = next_id
                next_id += 1

        nx.relabel_nodes(G, mapping, copy=False)
        # print(f"[INFO] Relabeled {len(mapping)} non-GPU nodes, total nodes now = {len(G.nodes)}")
        return mapping  # 可选：返回映射表，便于查
