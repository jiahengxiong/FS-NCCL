import copy
import itertools

import networkx as nx
import matplotlib.pyplot as plt
from utils import Network
from collections import defaultdict
from SIM.simulator import simulate_allgather_event_driven

# def build_nccl_ring_greedy_with_contention(G: nx.DiGraph, delay_attr="total_delay", target_nodes=None):
#     if target_nodes is None:
#         target_nodes = [n for n, d in G.nodes(data=True) if d.get("type") == "GPU"]
#
#     best_ring = None
#     best_max_segment_delay = float("inf")
#
#     for start_node in target_nodes:
#         unvisited = set(target_nodes)
#         unvisited.remove(start_node)
#         ring = [start_node]
#         current = start_node
#         segment_delays = []
#
#         while unvisited:
#             min_delay = float("inf")
#             next_node = None
#             best_path = None
#
#             for candidate in unvisited:
#                 try:
#                     path = nx.shortest_path(G, source=current, target=candidate)
#                     path_delay = sum(
#                         G[u][v]["transmission_delay"] + G[u][v]["propagation_delay"]
#                         for u, v in zip(path[:-1], path[1:])
#                     )
#
#                     if path_delay < min_delay:
#                         min_delay = path_delay
#                         next_node = candidate
#                         best_path = path
#
#                 except nx.NetworkXNoPath:
#                     continue
#
#             if next_node is None:
#                 break
#
#             segment_delays.append(min_delay)
#             ring.append(next_node)
#             unvisited.remove(next_node)
#             current = next_node
#
#         # Close the ring
#         try:
#             path = nx.shortest_path(G, source=current, target=start_node)
#             closing_delay = sum(
#                 G[u][v]["transmission_delay"] + G[u][v]["propagation_delay"]
#                 for u, v in zip(path[:-1], path[1:])
#             )
#             segment_delays.append(closing_delay)
#             ring.append(start_node)
#         except nx.NetworkXNoPath:
#             continue
#
#         max_seg = max(segment_delays)
#
#         if max_seg < best_max_segment_delay:
#             best_max_segment_delay = max_seg
#             best_ring = ring
#
#     return best_ring
#
# def extract_ring_subgraph(G: nx.DiGraph, ring_nodes: list, delay_attr="total_delay"):
#     ring_subgraph = nx.DiGraph()
#     added_edges = set()
#     for i in range(len(ring_nodes) - 1):
#         src = ring_nodes[i]
#         dst = ring_nodes[i + 1]
#         try:
#             path = nx.shortest_path(G, source=src, target=dst, weight=delay_attr)
#             for j in range(len(path) - 1):
#                 u, v = path[j], path[j + 1]
#                 if (u, v) not in added_edges:
#                     ring_subgraph.add_edge(u, v, **G[u][v])
#                     added_edges.add((u, v))
#         except (nx.NetworkXNoPath, nx.NetworkXError):
#             continue
#     return ring_subgraph

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


import networkx as nx

def build_ring(G, weight_attr="total_delay", wan_policy="edge-disjoint"):  # wan_policy: "node-disjoint" | "edge-disjoint" | "allow-overlap"
    """
    分三步构建最终的最小代价环：
      1) 在每个 DC 内部，求一个仅含 GPU 的“有向最小代价链”（Hamilton 路）；
         代价=相邻 GPU 间在该 DC 内的最短路径代价（禁止其它 GPU 作为中继）。
      2) 在全图上（禁止其它 GPU 作为中继），求两条跨 WAN 的最短路，将两个链首尾相连形成一个环；
         我们枚举两个链的方向（正/反）并选 WAN 连接总代价最小的组合。
         （wan_policy 可选："edge-disjoint"（默认，**两条桥不共享任何边**，WAN 节点可复用） | "node-disjoint"（两条桥的**中间节点**不复用） | "allow-overlap"（两条桥可重叠））。
      3) 返回“铺开”的底层有向图（包含 GPU、Border、Router 等所有实际节点与边）。

    ring.graph:
      - "gpu_order": 最终环上的 GPU 顺序
      - "chain_dc0": DC0 内部链（顺序）
      - "chain_dc1": DC1 内部链（顺序）
      - "bridge": {"dc0_tail->dc1_head": (u,v,cost), "dc1_tail->dc0_head": (u,v,cost)}
      - "total_cost": 总代价（两条 DC 内链段 + 两条 WAN 桥接段）
    """
    # ---- 基本集合 ----
    gpus_dc0 = sorted([n for n, d in G.nodes(data=True) if d.get("type")=="GPU" and d.get("field")=="DC0"], key=str)
    gpus_dc1 = sorted([n for n, d in G.nodes(data=True) if d.get("type")=="GPU" and d.get("field")=="DC1"], key=str)
    if not gpus_dc0 or not gpus_dc1:
        raise ValueError("DC0 与 DC1 都需要至少 1 个 GPU。")

    gpu_set = set(gpus_dc0 + gpus_dc1)

    # 权重
    def w(u, v, data):
        return data.get(weight_attr, 1.0)

    # ---- 约束下的最短路计算 ----
    def node_ok_dc(src, dst, dc_field):
        # 只允许同 DC 的节点，且禁止其它 GPU 作为中继
        def ok(n):
            if n in gpu_set and n not in (src, dst):
                return False
            return G.nodes[n].get("field") == dc_field or n in (src, dst)
        return ok

    def node_ok_cross(src, dst, exclude_nodes=None):
        # 全图允许，但禁止其它 GPU 作为中继；并可额外排除一批 WAN 中间节点
        if exclude_nodes is None:
            exclude_nodes = set()
        def ok(n):
            if n in exclude_nodes and n not in (src, dst):
                return False
            if n in gpu_set and n not in (src, dst):
                return False
            return True
        return ok

    def edge_ok_cross(exclude_edges=None):
        # 过滤掉已使用过的边（用于 edge-disjoint 策略）
        if exclude_edges is None:
            exclude_edges = set()
        def ok(u, v):
            return (u, v) not in exclude_edges
        return ok

    path_cache_dc = {}     # (dc, src, dst) -> (cost, path)
    path_cache_cross = {}  # (src, dst, sorted_nodes, sorted_edges) -> (cost, path)

    def shortest_dc(dc, src, dst):
        key = (dc, src, dst)
        if key in path_cache_dc:
            return path_cache_dc[key]
        H = nx.subgraph_view(G, filter_node=node_ok_dc(src, dst, dc))
        path = nx.shortest_path(H, source=src, target=dst, weight=w)
        cost = sum(G[u][v].get(weight_attr, 1.0) for u, v in zip(path, path[1:]))
        path_cache_dc[key] = (cost, path)
        return cost, path

    def shortest_cross(src, dst, exclude_nodes=None, exclude_edges=None):
        # Build a deterministic cache key without comparing ints vs strs
        nodes_key = tuple(sorted((str(n) for n in exclude_nodes))) if exclude_nodes else None
        edges_key = tuple(sorted((f"{str(u)}|{str(v)}" for (u, v) in exclude_edges))) if exclude_edges else None
        key = (str(src), str(dst), nodes_key, edges_key)
        if key in path_cache_cross:
            return path_cache_cross[key]
        H = nx.subgraph_view(
            G,
            filter_node=node_ok_cross(src, dst, exclude_nodes),
            filter_edge=edge_ok_cross(exclude_edges)
        )
        path = nx.shortest_path(H, source=src, target=dst, weight=w)
        cost = sum(G[u][v].get(weight_attr, 1.0) for u, v in zip(path, path[1:]))
        path_cache_cross[key] = (cost, path)
        return cost, path

    # ---- 求“DC 内最小代价链”（小规模直接全排列）----
    def best_chain(gpu_list, dc_field):
        if len(gpu_list) == 1:
            return gpu_list[:], 0.0, []  # order, cost, segs
        best = (float("inf"), None, None)  # cost, order, segs
        for perm in itertools.permutations(gpu_list):
            ok = True
            cost_sum = 0.0
            segs = []
            for a, b in zip(perm, perm[1:]):
                try:
                    c, p = shortest_dc(dc_field, a, b)
                except nx.NetworkXNoPath:
                    ok = False
                    break
                cost_sum += c
                segs.append((a, b, c, p))
            if ok and cost_sum < best[0]:
                best = (cost_sum, list(perm), segs)
        if best[1] is None:
            raise ValueError(f"{dc_field} 内部无法形成链，请检查连通性。")
        return best[1], best[0], best[2]

    order0, cost0, segs0 = best_chain(gpus_dc0, "DC0")
    order1, cost1, segs1 = best_chain(gpus_dc1, "DC1")

    # ---- 选择两条链方向 + WAN 桥接形成环（策略由 wan_policy 控制） ----
    def middle_nodes(path, endpoints):
        s, t = endpoints
        return [n for n in path[1:-1] if n not in (s, t)]

    def edges_of(path):
        return list(zip(path, path[1:]))

    def try_bridges(o0, o1, policy):
        head0, tail0 = o0[0], o0[-1]
        head1, tail1 = o1[0], o1[-1]

        def attempt(order_first):
            # order_first == "01" 表示先 tail0->head1，再 tail1->head0；反之亦然
            if order_first == "01":
                srcA, dstA = tail0, head1
                srcB, dstB = tail1, head0
            else:
                srcA, dstA = tail1, head0
                srcB, dstB = tail0, head1

            if policy == "node-disjoint":
                cA, pA = shortest_cross(srcA, dstA, exclude_nodes=None, exclude_edges=None)
                used_nodes = set(middle_nodes(pA, (srcA, dstA)))
                cB, pB = shortest_cross(srcB, dstB, exclude_nodes=used_nodes, exclude_edges=None)
            elif policy == "edge-disjoint":
                cA, pA = shortest_cross(srcA, dstA, exclude_nodes=None, exclude_edges=None)
                used_edges = set(edges_of(pA))
                cB, pB = shortest_cross(srcB, dstB, exclude_nodes=None, exclude_edges=used_edges)
            elif policy == "allow-overlap":
                cA, pA = shortest_cross(srcA, dstA, exclude_nodes=None, exclude_edges=None)
                cB, pB = shortest_cross(srcB, dstB, exclude_nodes=None, exclude_edges=None)
            else:
                raise ValueError("wan_policy must be one of: 'node-disjoint', 'edge-disjoint', 'allow-overlap'.")

            total = cA + cB
            # 返回以 0->1 的顺序为 p01/p10；若 order_first 是 "10" 则调换
            if order_first == "01":
                meta = ((srcA, dstA, cA), (srcB, dstB, cB))
                return total, pA, pB, meta
            else:
                meta = ((srcB, dstB, cB), (srcA, dstA, cA))
                return total, pB, pA, meta

        best = None
        for order_first in ("01", "10"):
            try:
                cand = attempt(order_first)
                if best is None or cand[0] < best[0]:
                    best = cand
            except nx.NetworkXNoPath:
                continue
        if best is None:
            raise nx.NetworkXNoPath("No WAN bridges possible under current policy.")
        return best  # (tot_bridge, p01, p10, meta)

    # 严格按用户策略尝试；若无解，直接抛错（确保“两条 WAN 路径”的约束生效）
    candidates = []
    for o0 in (order0, list(reversed(order0))):
        for o1 in (order1, list(reversed(order1))):
            try:
                tot_bridge, p01, p10, meta = try_bridges(o0, o1, wan_policy)
                candidates.append((tot_bridge, o0, o1, p01, p10, meta))
            except nx.NetworkXNoPath:
                continue

    if not candidates:
        raise ValueError(f"跨 WAN 无法把两个 DC 链串成环（策略: {wan_policy}）。")

    tot_bridge, final0, final1, p01, p10, meta = min(candidates, key=lambda x: x[0])
    chosen_policy = wan_policy

    # ---- 铺开为底层有向图 ----
    ring = nx.DiGraph()
    added = set()
    ordered_edges = []
    ordered_nodes = []

    def add_path(path):
        nonlocal ordered_nodes, ordered_edges
        if not ordered_nodes:
            ordered_nodes.extend(path)
        else:
            ordered_nodes.extend(path[1:])
        ordered_edges.extend(list(zip(path, path[1:])))
        for u, v in zip(path, path[1:]):
            if u not in ring:
                ring.add_node(u, **G.nodes[u])
            if v not in ring:
                ring.add_node(v, **G.nodes[v])
            if (u, v) not in added:
                ring.add_edge(u, v, **G.get_edge_data(u, v, default={}))
                added.add((u, v))

    # 铺 DC0
    segs0_final = []
    for a, b in zip(final0, final0[1:]):
        c, p = shortest_dc("DC0", a, b)
        segs0_final.append((a, b, c, p))
        add_path(p)
    # 桥 0->1
    add_path(p01)
    # 铺 DC1
    segs1_final = []
    for a, b in zip(final1, final1[1:]):
        c, p = shortest_dc("DC1", a, b)
        segs1_final.append((a, b, c, p))
        add_path(p)
    # 桥 1->0
    add_path(p10)

    total_cost = cost0 + cost1 + tot_bridge

    # --- 硬校验：GPU 不能作为任何中间节点 ---
    def assert_no_gpu_as_middle(path):
        for m in path[1:-1]:
            if G.nodes[m].get("type") == "GPU":
                raise AssertionError(f"GPU {m} 被用作中继节点，违反约束。")

    for _, _, _, p in segs0_final:
        assert_no_gpu_as_middle(p)
    for _, _, _, p in segs1_final:
        assert_no_gpu_as_middle(p)
    assert_no_gpu_as_middle(p01)
    assert_no_gpu_as_middle(p10)

    # 记录并校验 WAN 两条路径满足策略
    ring.graph["bridges_paths"] = {"dc0_tail->dc1_head": p01, "dc1_tail->dc0_head": p10}
    if chosen_policy == "edge-disjoint":
        set01 = set(zip(p01, p01[1:]))
        set10 = set(zip(p10, p10[1:]))
        assert set01.isdisjoint(set10), "edge-disjoint 违反：两条 WAN 桥共享了边"
    elif chosen_policy == "node-disjoint":
        mids01 = set(p01[1:-1])
        mids10 = set(p10[1:-1])
        assert mids01.isdisjoint(mids10), "node-disjoint 违反：两条 WAN 桥共享了中间节点"

    ring.graph["gpu_order"] = final0 + final1
    ring.graph["chain_dc0"] = final0
    ring.graph["chain_dc1"] = final1
    ring.graph["chain_dc0_segs"] = [(a, b, c) for (a, b, c, _) in segs0_final]
    ring.graph["chain_dc1_segs"] = [(a, b, c) for (a, b, c, _) in segs1_final]
    ring.graph["bridge"] = {
        "dc0_tail->dc1_head": meta[0],
        "dc1_tail->dc0_head": meta[1],
    }
    ring.graph["edge_sequence"] = ordered_edges
    ring.graph["node_sequence"] = ordered_nodes
    ring.graph["total_cost"] = total_cost
    ring.graph["wan_policy"] = chosen_policy
    ring.graph["weight_attr"] = weight_attr
    return ring

import networkx as nx
import copy

def reconfigure_WAN(G, ring, weight_attr="total_delay"):
    """
    仅重配 WAN：
      - 保留 ring 中的两条 DC 内 GPU→GPU chain（删除其余边）
      - 在 G 上求两条最短 WAN 路把两条 chain 串成 ring（禁止其它 GPU 作为中继）
      - 将两条 WAN 路的所有边（含属性）加入到 AG
    返回:
      AG: 只优化了 WAN 部分后的有向 ring 子图
    """
    # --- 帮助函数 ---
    def is_gpu(n):
        return G.nodes[n].get("type") == "GPU"

    def w(u, v, data):
        return data.get(weight_attr, None)

    # 1) 拷贝 + 去掉所有非 GPU→GPU 的边，只保留两条 DC 内的 GPU 链
    AG = copy.copy(ring)
    to_remove = []
    for u, v in AG.edges():
        # 只保留 GPU→GPU 的链段；其它边（包括旧 WAN）全部移除
        if not (is_gpu(u) or is_gpu(v)):
            to_remove.append((u, v))
    AG.remove_edges_from(to_remove)

    # 2) 在 GPU 子图中识别两条有向 chain（head: in=0,out=1；tail: in=1,out=0）
    gpu_nodes = [n for n in AG.nodes() if is_gpu(n)]
    H = AG.subgraph(gpu_nodes).copy()

    # 找每个连通组件的 head / tail 并还原顺序
    chains = []
    # 将无向连通分量分开，再按出度/入度恢复有向链顺序
    for comp in nx.weakly_connected_components(H):
        sub = H.subgraph(comp).copy()
        if sub.number_of_nodes() == 0:
            continue
        # head: in_degree==0 的唯一点；tail: out_degree==0 的唯一点
        heads = [n for n in sub.nodes() if sub.in_degree(n) == 0 and sub.out_degree(n) == 1]
        tails = [n for n in sub.nodes() if sub.out_degree(n) == 0 and sub.in_degree(n) == 1]
        # 退化成一个点的链
        if len(sub) == 1:
            only = next(iter(sub.nodes()))
            chains.append([only])
            continue
        if len(heads) != 1 or len(tails) != 1:
            raise ValueError(f"GPU 子图不是两条干净的有向链（检测到异常头/尾）。heads={heads}, tails={tails}")
        head, tail = heads[0], tails[0]
        # 顺着 out-edge 走到尾
        order = [head]
        cur = head
        while cur != tail:
            succs = list(sub.successors(cur))
            if len(succs) != 1:
                raise ValueError("链上出现分叉，预期每个中间 GPU 的出度为 1。")
            cur = succs[0]
            order.append(cur)
        chains.append(order)

    if len(chains) != 2:
        raise ValueError(f"期望恰好两条 GPU 链，但检测到 {len(chains)} 条。")

    # 3) 在 G 上为两条链求 WAN 最短路：tail0->head1 与 tail1->head0（禁止其它 GPU 作为中继）
    (chain0, chain1) = chains
    head0, tail0 = chain0[0], chain0[-1]
    head1, tail1 = chain1[0], chain1[-1]

    gpu_set = set([n for n in G.nodes() if is_gpu(n)])

    def node_ok(src, dst):
        # 禁止任何其它 GPU 作为中继；端点放行
        def ok(n):
            if n in gpu_set and n not in (src, dst):
                return False
            return True
        return ok

    def shortest_no_mid_gpu(src, dst):
        Hview = nx.subgraph_view(G, filter_node=node_ok(src, dst))
        path = nx.shortest_path(Hview, source=src, target=dst, weight=w)
        return path

    try:
        path01 = shortest_no_mid_gpu(tail0, head1)  # 链0尾 → 链1头
    except nx.NetworkXNoPath:
        raise ValueError(f"WAN 不可达：{tail0} → {head1}")
    try:
        path10 = shortest_no_mid_gpu(tail1, head0)  # 链1尾 → 链0头
    except nx.NetworkXNoPath:
        raise ValueError(f"WAN 不可达：{tail1} → {head0}")

    # 4) 把两条 WAN 路的所有节点与边（含属性）加入到 AG
    def add_path_edges(subgraph, path):
        # 先确保所有节点在图里，属性继承自 G
        for n in path:
            if n not in subgraph:
                subgraph.add_node(n, **G.nodes[n])
        # 再逐边加入，继承原图所有边属性（注意你的图是多边图 key=uuid；这里合并为一条逻辑边）
        for u, v in zip(path, path[1:]):
            # 如果原图是 MultiDiGraph，这里可以聚合/选最小 cost 一条；当前是 DiGraph，直接取属性
            data = G.get_edge_data(u, v, default={})
            subgraph.add_edge(u, v, **data)

    add_path_edges(AG, path01)
    add_path_edges(AG, path10)

    # —— 用“链 + WAN 路”确定性地构建有序环序列 ——
    # 构建链的边序列（只在 GPU 子图上，链内必然存在这些边）
    def chain_edges(chain_order):
        return list(zip(chain_order, chain_order[1:]))

    def path_edges(path):
        return list(zip(path, path[1:]))

    # 期望连接关系： chain0_tail == path01[0], path01[-1] == chain1_head,
    #               chain1_tail == path10[0], path10[-1] == chain0_head
    assert chain0[-1] == path01[0], "链0 尾与 WAN 路 0->1 的起点不一致"
    assert path01[-1] == chain1[0], "WAN 路 0->1 的终点不是链1 头"
    assert chain1[-1] == path10[0], "链1 尾与 WAN 路 1->0 的起点不一致"
    assert path10[-1] == chain0[0], "WAN 路 1->0 的终点不是链0 头"

    ordered_edges = []
    ordered_edges += chain_edges(chain0)
    ordered_edges += path_edges(path01)
    ordered_edges += chain_edges(chain1)
    ordered_edges += path_edges(path10)

    ordered_nodes = [ordered_edges[0][0]] + [v for (_, v) in ordered_edges]

    AG.graph["edge_sequence"] = ordered_edges
    AG.graph["node_sequence"] = ordered_nodes

    # 可选：把结果顺序存在 graph 属性里，方便你打印验证
    AG.graph["chains"] = {"DC0_or_A": chain0, "DC1_or_B": chain1}
    AG.graph["wan_paths"] = {"tail0->head1": path01, "tail1->head0": path10}
    AG.graph["weight_attr"] = weight_attr

    return AG



def reconfigure_CCL(G: nx.DiGraph, ring: nx.DiGraph, weight_attr="total_delay"):
    """
    根据当前 G（全局网络）与 ring（旧 CCL 环），重构一个只优化 CCL 部分的新环。
    规则：
      - GPU 相关边（任一端为 GPU）从 G 中保留；
      - WAN 内部边（两端均非 GPU）从 ring 中保留；
      - 节点与边的所有属性都会完整复制；
      - 最后在合并图 AG 上重新构建 ring。
    """
    AG = nx.DiGraph()

    # --- 添加节点和边：GPU 相关边来自 G ---
    for (u, v, data) in G.edges(data=True):
        if G.nodes[u].get("type") == "GPU" or G.nodes[v].get("type") == "GPU":
            # 确保两端节点存在且复制属性
            if u not in AG:
                AG.add_node(u, **G.nodes[u])
            if v not in AG:
                AG.add_node(v, **G.nodes[v])
            AG.add_edge(u, v, **data)

    # --- 添加 WAN 内部边：来自 ring ---
    for (u, v, data) in ring.edges(data=True):
        if G.nodes[u].get("type") != "GPU" and G.nodes[v].get("type") != "GPU":
            if u not in AG:
                AG.add_node(u, **G.nodes[u])
            if v not in AG:
                AG.add_node(v, **G.nodes[v])
            AG.add_edge(u, v, **data)

    # --- 调用 build_ring 在 AG 上构建新的环 ---
    new_ring = build_ring(AG, weight_attr=weight_attr)
    return new_ring

def max_gpu_segment_cost_along_ring(ring: nx.DiGraph, weight_attr="total_delay"):
    """
    沿 ring 方向，计算每个 GPU -> 下一 GPU 段的代价（边权累加），返回最大段代价（标量）。
    代价优先使用 edge[weight_attr]；若缺失且有 transmission_delay/propagation_delay，则用两者之和兜底。
    """

    def edge_cost(u, v):
        data = ring.get_edge_data(u, v)
        if data is None:
            raise ValueError(f"环中缺少边 {u} -> {v} 的属性。")
        if weight_attr in data:
            return float(data[weight_attr])
        td = data.get("transmission_delay")
        pd = data.get("propagation_delay")
        if td is not None and pd is not None:
            return float(td) + float(pd)
        raise ValueError(f"边 {u}->{v} 缺少 '{weight_attr}'，且没有 (transmission_delay, propagation_delay) 可兜底。")

    # 1) 仅使用明确的 edge_sequence（环可能经过 WAN，非简单出度不可依赖“唯一后继”）
    ordered_edges = ring.graph.get("edge_sequence", None)
    if not ordered_edges:
        raise ValueError("ring.graph['edge_sequence'] 缺失，无法确定环方向。")
    if not all(ring.has_edge(u, v) for (u, v) in ordered_edges):
        missing = [(u, v) for (u, v) in ordered_edges if not ring.has_edge(u, v)]
        raise ValueError(f"edge_sequence 中存在不在 ring 里的边：{missing[:3]} ...")

    nodes_seq = [ordered_edges[0][0]] + [v for (_, v) in ordered_edges]
    if nodes_seq[0] != nodes_seq[-1]:
        nodes_seq.append(nodes_seq[0])

    # 2) 找出序列里 GPU 的索引
    nodes_linear = nodes_seq[:-1]  # 去掉闭合的最后一个重复起点
    gpu_idxs = [i for i, n in enumerate(nodes_linear) if ring.nodes[n].get("type") == "GPU"]
    if len(gpu_idxs) < 2:
        raise ValueError("GPU 数不足以构成环段。")

    # 3) 逐段累加边权，取最大
    L = len(nodes_linear)
    max_cost = 0.0
    for k in range(len(gpu_idxs)):
        i = gpu_idxs[k]
        j = gpu_idxs[(k + 1) % len(gpu_idxs)]
        # 从 i 沿环走到 j
        cost = 0.0
        pos = i
        while True:
            u = nodes_linear[pos]
            v = nodes_linear[(pos + 1) % L]
            cost += edge_cost(u, v)
            pos = (pos + 1) % L
            if pos == j:
                break
        if cost > max_cost:
            max_cost = cost

    return max_cost
def update_ring_edges_from_G(G, ring):
    """
    将 G 上所有边的属性同步更新到 ring 上。
    假设 ring 的边集合是 G 的子集。
    """
    updated_edges = 0
    missing_edges = []

    for u, v in ring.edges():
        if G.has_edge(u, v):
            # 拿到 G 中的属性
            attrs = G.edges[u, v]['total_delay']
            if attrs is None:
                print("ERROR!!! NO TOTAL DELAY!!!")
                continue
            # 清空 ring 中的旧属性后更新
            ring.edges[u, v]["total_delay"] = attrs
            updated_edges += 1
        else:
            missing_edges.append((u, v))


    if missing_edges:
        print(f"[Warning] {len(missing_edges)} edges in ring not found in G: {missing_edges}")
    # else:
    #     print(f"[Info] Updated {updated_edges} edges from G to ring.")

    return ring

def fastslow(packet_size):
    collective = 100000
    network = Network.Network()
    network.build_link_intraDC()
    G = network.topology
    ring_set = []
    for u, v, data in G.edges(data=True):
        capacity = data.get("capacity", None)
        if capacity is not None and capacity > 0:
            delay = packet_size / capacity
            data["transmission_delay"] = delay
            data['total_delay'] = delay + data['propagation_delay']
        else:
            data["transmission_delay"] = float("inf")
            data["total_delay"] = float("inf")
    ring = build_ring(G, wan_policy="edge-disjoint", weight_attr="total_delay")  # 可选: 'edge-disjoint' | 'node-disjoint' | 'allow-overlap'
    for edge in ring.edges():
        print(edge)
    simulate_allgather_event_driven(ring, verbose=True)


    collective_time = 0
    for c in range(collective):
        network = Network.Network()
        network.build_link_intraDC()
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

        ring = update_ring_edges_from_G(G, ring)
        # ring = reconfigure_WAN(G, ring, weight_attr="total_delay")
        if collective%10==0:
            ring = build_ring(G)
            ring = update_ring_edges_from_G(G, ring)
        else:
            ring = update_ring_edges_from_G(G, ring)
            ring = reconfigure_WAN(G, ring, weight_attr="total_delay")
        ring = update_ring_edges_from_G(G, ring)

        collective_time += simulate_allgather_event_driven(ring, verbose=False)


    print("collective time:", collective_time)
    return collective_time



if __name__ == '__main__':
    packet_size = 64/ 1024  # bits
    fastslow(packet_size)