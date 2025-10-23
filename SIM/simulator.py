import heapq
import itertools
import networkx as nx

def extract_gpu_segments(ring: nx.DiGraph):
    """
    仅依赖 ring.graph['edge_sequence']（环上按方向展开的一圈边），
    为每个 GPU 提取它到“下一 GPU”的底层边序列。
    返回:
      gpu_order: [gpu0, gpu1, ...]  按环方向的 GPU 顺序
      segment_edges: dict[gpu_i] = [(u,v), (u,v), ...]  # 从 gpu_i 到下一 GPU 的唯一边序列
      nodes_linear: 展开的节点序列（去掉闭合终点）
    """
    ordered_edges = ring.graph.get("edge_sequence")
    if not ordered_edges:
        raise ValueError("缺少 ring.graph['edge_sequence']，无法确定环方向。")
    if not all(ring.has_edge(u, v) for (u, v) in ordered_edges):
        missing = [(u, v) for (u, v) in ordered_edges if not ring.has_edge(u, v)]
        raise ValueError(f"edge_sequence 中有不在 ring 的边，例如 {missing[:3]}")

    nodes_seq = [ordered_edges[0][0]] + [v for (_, v) in ordered_edges]
    if nodes_seq[0] != nodes_seq[-1]:
        nodes_seq.append(nodes_seq[0])
    nodes_linear = nodes_seq[:-1]  # 去掉闭合的最后一个点

    # 环上 GPU 顺序（严格按出现顺序）
    gpu_idxs = [i for i, n in enumerate(nodes_linear) if ring.nodes[n].get("type") == "GPU"]
    if len(gpu_idxs) < 2:
        raise ValueError("GPU 数不足以形成环。")
    gpu_order = [nodes_linear[i] for i in gpu_idxs]

    # 为每个 GPU 抽出它到下一 GPU 的边序列（只按 nodes_linear 的索引走，不看图的出度）
    L = len(nodes_linear)
    segment_edges = {}
    for k in range(len(gpu_idxs)):
        i = gpu_idxs[k]
        j = gpu_idxs[(k + 1) % len(gpu_idxs)]
        edges = []
        pos = i
        while True:
            u = nodes_linear[pos]
            v = nodes_linear[(pos + 1) % L]
            edges.append((u, v))
            pos = (pos + 1) % L
            if pos == j:
                break
        segment_edges[gpu_order[k]] = edges

    return gpu_order, segment_edges, nodes_linear


def simulate_allgather_event_driven(ring: nx.DiGraph, weight_attr="total_delay", verbose=False, node_serialize_tx=True, duplex="full"):
    """
    事件驱动 + 流水线 的 Ring All-Gather 模拟器。
    - 不依赖图的“唯一后继”假设；严格使用 extract_gpu_segments() 得到的固定传输路径。
    - 每条底层边 (u,v) 有自己的 busy_until（资源占用），不同段共享边时会排队。
    - 边时延优先取 edge[weight_attr]；缺失则用 transmission_delay+propagation_delay。
    返回:
      finish_time: float
      detail: {
        "gpu_order": [...],
        "segment_edges": {gpu: [(u,v), ...]},
        "edge_delay": {(u,v): delay},
        "edge_busy_until": {(u,v): time},
        "receives": {(gpu, chunk_owner): time},
      }
    """
    gpu_order, segment_edges, _ = extract_gpu_segments(ring)
    P = len(gpu_order)
    if verbose:
        print(f"[Init] GPU order: {gpu_order}")

    # 读取边时延
    def edge_delay(u, v):
        data = ring.get_edge_data(u, v)
        if data is None:
            raise ValueError(f"环中缺少边 {u}->{v} 的属性")
        if weight_attr in data:
            return float(data[weight_attr])
        td = data.get("transmission_delay")
        pd = data.get("propagation_delay")
        if td is None or pd is None:
            raise ValueError(f"边 {u}->{v} 缺少 '{weight_attr}' 且无 (transmission_delay, propagation_delay) 兜底")
        return float(td) + float(pd)

    # 收集本次环上用到的所有边
    used_edges = set()
    for edges in segment_edges.values():
        used_edges.update(edges)
    edge_w = {e: edge_delay(*e) for e in used_edges}
    edge_busy_until = {e: 0.0 for e in used_edges}

    # 节点发送/接收占用（节点级真实事件）：
    nodes_in_use = set()
    for (u, v) in used_edges:
        nodes_in_use.add(u); nodes_in_use.add(v)
    node_tx_busy_until = {n: 0.0 for n in nodes_in_use}
    node_rx_busy_until = {n: 0.0 for n in nodes_in_use}  # 仅在半双工下使用

    # 事件日志（边级与节点级）
    event_log = []  # tuples like (kind, t_start, t_end, info_dict)

    def is_wan_edge(e):
        u, v = e
        fu = ring.nodes[u].get("field")
        fv = ring.nodes[v].get("field")
        return fu == "WAN" or fv == "WAN"

    # 环邻接（GPU层）
    next_gpu = {gpu_order[i]: gpu_order[(i + 1) % P] for i in range(P)}

    # 节点何时“拥有”某个 chunk（到达即可转发）
    has_time = {(gpu, chunk): (0.0 if gpu == chunk else float("inf"))
                for gpu in gpu_order for chunk in gpu_order}
    # 避免重复发送同一 (node, chunk)
    forwarded = {(gpu, chunk): False for gpu in gpu_order for chunk in gpu_order}

    # 统计接收
    recv_time = {}  # (gpu, chunk_owner) -> time
    total_receives_needed = P * (P - 1)
    received = 0
    finish_time = 0.0

    # 事件队列： (time, seq, type, a, b)
    # type: "SEND" (src_gpu, chunk_owner)  or  "ARRIVE" (dst_gpu, chunk_owner)
    pq = []
    counter = itertools.count()

    def push(evt_type, t, a, b):
        heapq.heappush(pq, (t, next(counter), evt_type, a, b))

    # 初始：每个 GPU 在 t=0 发送自己的 chunk 给下一 GPU
    for g in gpu_order:
        push("SEND", 0.0, g, g)

    while pq and received < total_receives_needed:
        t_now, _, evt, X, Y = heapq.heappop(pq)
        if evt == "SEND":
            src, chunk = X, Y
            if forwarded[(src, chunk)]:
                continue
            # 如果该 chunk 尚未到达 src，延迟到可用时刻再发
            t_ready = has_time[(src, chunk)]
            if t_now < t_ready:
                push("SEND", t_ready, src, chunk)
                continue

            path_edges = segment_edges[src]  # 唯一固定路径
            if verbose:
                print(f"[SEND] t={t_now:.6f}, src={src}, chunk={chunk}, path_len={len(path_edges)}")

            cur_t = t_now
            for e in path_edges:
                u, v = e
                # 计算这条边可以开始的时间：受 (1) 当前消息到达该节点的时间 cur_t，
                # (2) 边是否空闲 edge_busy_until[e]，(3) 节点是否可发送 node_tx_busy_until[u]，
                # (4) 若半双工，还要等接收口空闲 node_rx_busy_until[u]
                t_link_ready = edge_busy_until[e]
                t_tx_ready = node_tx_busy_until[u] if node_serialize_tx else 0.0
                t_half_rx_ready = node_rx_busy_until[u] if (duplex == "half") else 0.0
                start_e = max(cur_t, t_link_ready, t_tx_ready, t_half_rx_ready)

                # 打印/记录等待原因
                if verbose and start_e > cur_t:
                    waits = []
                    if start_e == t_link_ready and t_link_ready > cur_t:
                        waits.append("edge")
                    if start_e == t_tx_ready and t_tx_ready > cur_t and node_serialize_tx:
                        waits.append("node_tx")
                    if duplex == "half" and start_e == t_half_rx_ready and t_half_rx_ready > cur_t:
                        waits.append("node_rx(half)")
                    if verbose:
                        print(f"  [WAIT] at {u} for {u}->{v} until {start_e:.6f} due to {','.join(waits)}")

                finish_e = start_e + edge_w[e]

                # 更新资源占用
                edge_busy_until[e] = finish_e
                if node_serialize_tx:
                    node_tx_busy_until[u] = finish_e
                if duplex == "half":
                    # 半双工：发送期间该节点的接收口也被占用
                    node_rx_busy_until[u] = finish_e
                # 记录事件
                event_log.append((
                    "LINK", start_e, finish_e,
                    {"edge": e, "src": src, "dst": v, "chunk": chunk}
                ))
                if verbose:
                    print(f"  [LINK] {u}->{v} start={start_e:.6f} finish={finish_e:.6f}")
                cur_t = finish_e

            dst = next_gpu[src]
            forwarded[(src, chunk)] = True
            push("ARRIVE", cur_t, dst, chunk)

        elif evt == "ARRIVE":
            dst, chunk = X, Y
            # 记录接收（自己的 chunk 不算接收）
            if dst != chunk and (dst, chunk) not in recv_time:
                recv_time[(dst, chunk)] = t_now
                received += 1
                if t_now > finish_time:
                    finish_time = t_now
            if verbose:
                print(f"[ARRIVE] t={t_now:.6f}, dst={dst}, chunk={chunk}, received={received}/{total_receives_needed}")
            # 到达即拥有，可立刻转发
            if t_now < has_time[(dst, chunk)]:
                has_time[(dst, chunk)] = t_now
            # 节点接收口占用：到达时刻视为接收已完成，若半双工，短暂占用到 t_now（可选更复杂模型）
            if duplex == "half" and node_rx_busy_until.get(dst, 0.0) < t_now:
                node_rx_busy_until[dst] = t_now
            event_log.append((
                "ARRIVE", t_now, t_now,
                {"node": dst, "chunk": chunk}
            ))
            # 还没回到 chunk 的原主，则继续沿固定路径转发
            if dst != chunk and not forwarded[(dst, chunk)]:
                push("SEND", t_now, dst, chunk)

    # --- Sanity checks ---
    P = len(gpu_order)
    expected = P * (P - 1)
    if received != expected:
        missing = []
        for g in gpu_order:
            for c in gpu_order:
                if g == c:
                    continue
                if (g, c) not in recv_time:
                    missing.append((g, c))
        raise RuntimeError(f"接收计数不一致: got {received}, expected {expected}. Missing samples: {missing[:8]} ...")

    # WAN utilization summary
    wan_used = {e: t for e, t in edge_busy_until.items() if is_wan_edge(e) and t > 0.0}
    if verbose:
        print(f"[FINISH] All-Gather done at t={finish_time:.6f} | WAN edges used: {len(wan_used)} / {len(edge_busy_until)}")

    detail = {
        "gpu_order": gpu_order,
        "segment_edges": segment_edges,
        "edge_delay": edge_w,
        "edge_busy_until": edge_busy_until,
        "receives": recv_time,
        "node_tx_busy_until": node_tx_busy_until,
        "node_rx_busy_until": node_rx_busy_until,
        "events": event_log,
    }
    return finish_time