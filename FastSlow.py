import networkx as nx
import matplotlib.pyplot as plt
from utils import Network
from collections import defaultdict

def build_nccl_ring_greedy_with_contention(G: nx.DiGraph, delay_attr="total_delay", target_nodes=None):
    if target_nodes is None:
        target_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "GPU"]

    best_ring = None
    best_max_segment_delay = float("inf")

    for start_node in target_nodes:
        unvisited = set(target_nodes)
        unvisited.remove(start_node)
        ring = [start_node]
        current = start_node
        segment_delays = []

        while unvisited:
            min_delay = float("inf")
            next_node = None
            best_path = None

            for candidate in unvisited:
                try:
                    path = nx.shortest_path(G, source=current, target=candidate)
                    path_delay = sum(
                        G[u][v]["transmission_delay"] + G[u][v]["propagation_delay"]
                        for u, v in zip(path[:-1], path[1:])
                    )

                    if path_delay < min_delay:
                        min_delay = path_delay
                        next_node = candidate
                        best_path = path

                except nx.NetworkXNoPath:
                    continue

            if next_node is None:
                break

            segment_delays.append(min_delay)
            ring.append(next_node)
            unvisited.remove(next_node)
            current = next_node

        # Close the ring
        try:
            path = nx.shortest_path(G, source=current, target=start_node)
            closing_delay = sum(
                G[u][v]["transmission_delay"] + G[u][v]["propagation_delay"]
                for u, v in zip(path[:-1], path[1:])
            )
            segment_delays.append(closing_delay)
            ring.append(start_node)
        except nx.NetworkXNoPath:
            continue

        max_seg = max(segment_delays)

        if max_seg < best_max_segment_delay:
            best_max_segment_delay = max_seg
            best_ring = ring

    return best_ring

def extract_ring_subgraph(G: nx.DiGraph, ring_nodes: list, delay_attr="total_delay"):
    ring_subgraph = nx.DiGraph()
    added_edges = set()
    for i in range(len(ring_nodes) - 1):
        src = ring_nodes[i]
        dst = ring_nodes[i + 1]
        try:
            path = nx.shortest_path(G, source=src, target=dst, weight=delay_attr)
            for j in range(len(path) - 1):
                u, v = path[j], path[j + 1]
                if (u, v) not in added_edges:
                    ring_subgraph.add_edge(u, v, **G[u][v])
                    added_edges.add((u, v))
        except (nx.NetworkXNoPath, nx.NetworkXError):
            continue
    return ring_subgraph

def draw_ring_subgraph(ring_subgraph: nx.DiGraph, delay_attr="total_delay"):
    pos = nx.spring_layout(ring_subgraph, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx_nodes(ring_subgraph, pos, node_size=800, node_color='lightblue')
    nx.draw_networkx_labels(ring_subgraph, pos, font_size=10)
    for u, v, data in ring_subgraph.edges(data=True):
        label = f"{data.get(delay_attr, '?'):.1e}"
        nx.draw_networkx_edges(ring_subgraph, pos, edgelist=[(u, v)], connectionstyle="arc3,rad=0.1", width=1.5)
        nx.draw_networkx_edge_labels(ring_subgraph, pos, edge_labels={(u, v): label}, font_size=8)
    plt.title(f"Ring Subgraph with Edge '{delay_attr}' Annotated")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main(packet_size):
    collective = 100000
    network = Network.Network()
    G = network.topology
    for u, v, data in G.edges(data=True):
        capacity = data.get("capacity", None)
        if capacity is not None and capacity > 0:
            delay = packet_size / capacity
            data["transmission_delay"] = delay
            data['total_delay'] = delay + data['propagation_delay']
        else:
            data["transmission_delay"] = float("inf")
            data["total_delay"] = float("inf")
    ring_nodes = build_nccl_ring_greedy_with_contention(G)
    # print("构造的逻辑 Ring 节点顺序为:", ring_nodes)
    ring_subgraph = extract_ring_subgraph(G, ring_nodes)
    # draw_ring_subgraph(ring_subgraph)
    collective_time = 0
    for i in range(0, collective):
        if i % 10 == 0:
            ring_nodes = build_nccl_ring_greedy_with_contention(G)
            # print("构造的逻辑 Ring 节点顺序为:", ring_nodes)
            ring_subgraph = extract_ring_subgraph(G, ring_nodes)
        else:
            AG = nx.DiGraph()
            AG.add_nodes_from(G.nodes(data=True))
            for u, v, data in G.edges(data=True):
                # print(u, v, G.nodes[u], G.nodes[v])
                if G.nodes[u]['type'] != 'GPU' and G.nodes[v]['type'] != 'GPU':
                    AG.add_edge(u, v, **data)
                else:
                    if ring_subgraph.has_edge(u, v):
                        AG.add_edge(u, v, **data)
            ring_nodes = build_nccl_ring_greedy_with_contention(AG)
            # print("构造的逻辑 Ring 节点顺序为:", ring_nodes)
            ring_subgraph = extract_ring_subgraph(AG, ring_nodes)
            # draw_ring_subgraph(AG, delay_attr="total_delay")


        for u, v, data in ring_subgraph.edges(data=True):
            ring_subgraph.edges[u, v]["total_delay"] = G.edges[u, v]["total_delay"]


        total_delay_list = []
        for j in range(len(ring_nodes) - 1):
            src = ring_nodes[j]
            dst = ring_nodes[j+1]
            cost = nx.shortest_path_length(ring_subgraph, source=src, target=dst, weight="total_delay")
            total_delay_list.append(cost)
        min_delay = max(total_delay_list)
        collective_time += min_delay * 7
    print(collective_time)

if __name__ == '__main__':
    packet_size = 64 / 1024  # bits
    main(packet_size)