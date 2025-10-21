import networkx as nx
import uuid
from utils.Traffic import get_bandwidth_trace


class Network:
    def __init__(self):
        self.topology = self.get_topology()

    def get_topology(self):
        G = nx.DiGraph()
        G.add_node(0, type='GPU')
        G.add_node(1, type='GPU')
        G.add_node(2, type='GPU')
        G.add_node(3, type='GPU')
        G.add_node(4, type='GPU')
        G.add_node(5, type='GPU')
        G.add_node(6, type='GPU')
        G.add_node(7, type='GPU')

        G.add_node('Border Switch 0', type='Boarder Switch')
        G.add_node('Border Switch 1', type='Boarder Switch')
        G.add_node('Border Switch 2', type='Boarder Switch')
        G.add_node('Border Switch 3', type='Boarder Switch')

        G.add_node('Router 0', type='Router')
        G.add_node('Router 1', type='Router')
        G.add_node('Router 2', type='Router')
        G.add_node('Router 3', type='Router')
        G.add_node('Router 4', type='Router')

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
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100),  propagation_delay=1e-5)
        G.add_edge('Router 0', 'Border Switch 0', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100),  propagation_delay=1e-5)
        G.add_edge('Border Switch 1', 'Router 0', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100),  propagation_delay=1e-5)
        G.add_edge('Router 0', 'Border Switch 1', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100),  propagation_delay=1e-5)


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
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100),  propagation_delay=1e-5)
        G.add_edge('Router 3', 'Border Switch 2', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100),  propagation_delay=1e-5)
        G.add_edge('Border Switch 3', 'Router 3', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100),  propagation_delay=1e-5)
        G.add_edge('Router 3', 'Border Switch 3', key=uuid.uuid4().hex,
                   capacity=get_bandwidth_trace(method='gaussian', steps=1, capacity=100),  propagation_delay=1e-5)




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


        return G
