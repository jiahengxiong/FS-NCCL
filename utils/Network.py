import networkx as nx
import uuid
from utils.Traffic import get_bandwidth_trace


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

        G.add_edge(0, 'Border Switch 0', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 0', 0, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(0, 'Border Switch 1', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 1', 0, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(1, 'Border Switch 0', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 0', 1, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(1, 'Border Switch 1', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 1', 1, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(2, 'Border Switch 0', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 0', 2, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(2, 'Border Switch 1', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 1', 2, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(3, 'Border Switch 0', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 0', 3, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(3, 'Border Switch 1', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 1', 3, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge('Border Switch 0', 'Router 0', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 0', 'Border Switch 0', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Border Switch 1', 'Router 0', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 0', 'Border Switch 1', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)

        G.add_edge('Router 0', 'Router 1', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 1', 'Router 0', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 0', 'Router 4', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 4', 'Router 0', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 1', 'Router 2', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 2', 'Router 1', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 2', 'Router 3', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 3', 'Router 2', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 2', 'Router 4', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 4', 'Router 2', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 3', 'Router 4', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 4', 'Router 3', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)

        G.add_edge('Border Switch 2', 'Router 3', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 3', 'Border Switch 2', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Border Switch 3', 'Router 3', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)
        G.add_edge('Router 3', 'Border Switch 3', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100), propagation_delay=1e-5)

        G.add_edge(4, 'Border Switch 2', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 2', 4, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(4, 'Border Switch 3', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 3', 4, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(5, 'Border Switch 2', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 2', 5, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(5, 'Border Switch 3', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 3', 5, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(6, 'Border Switch 2', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 2', 6, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(6, 'Border Switch 3', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 3', 6, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(7, 'Border Switch 2', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 2', 7, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)

        G.add_edge(7, 'Border Switch 3', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
        G.add_edge('Border Switch 3', 7, key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200), propagation_delay=7e-7)
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
                    self.topology.add_edge(DC_0_GPU_list[i], DC_0_GPU_list[j], key=uuid.uuid4().hex,
                                           capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200),
                                           propagation_delay=7e-7)
                    self.topology.add_edge(DC_0_GPU_list[j], DC_0_GPU_list[i], key=uuid.uuid4().hex,
                                           capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200),
                                           propagation_delay=7e-7)

        DC_1_GPU_list = []
        for node in self.topology.nodes():
            if self.topology.nodes[node]['type'] == 'GPU' and self.topology.nodes[node]['field'] == 'DC1':
                DC_1_GPU_list.append(node)
        for i in range(len(DC_1_GPU_list)):
            for j in range(i + 1, len(DC_1_GPU_list)):
                if i != j:
                    self.topology.add_edge(DC_1_GPU_list[i], DC_1_GPU_list[j], key=uuid.uuid4().hex,
                                           capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200),
                                           propagation_delay=7e-7)
                    self.topology.add_edge(DC_1_GPU_list[j], DC_1_GPU_list[i], key=uuid.uuid4().hex,
                                           capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=200),
                                           propagation_delay=7e-7)

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
