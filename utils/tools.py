import networkx as nx

def relabel_ring_gpus_simple(ring: nx.DiGraph):
    """
    从 ring.graph['edge_sequence'] 按顺序提取节点，
    去掉非 GPU 节点，只保留 GPU 顺序；
    然后将这些 GPU 按出现顺序重新命名为 0,1,2,...。
    其它节点（WAN、Router 等）保持原名。
    """
    edges = ring.graph.get("edge_sequence")
    if not edges:
        raise ValueError("ring.graph 缺少 edge_sequence。")

    # 从 edge_sequence 里摊平节点序列
    nodes_seq = [edges[0][0]] + [v for (_, v) in edges]
    if nodes_seq[0] != nodes_seq[-1]:
        nodes_seq.append(nodes_seq[0])

    # 只保留 GPU 节点（保持首次出现顺序）
    gpu_order, seen = [], set()
    for n in nodes_seq:
        if ring.nodes[n].get("type") == "GPU" and n not in seen:
            seen.add(n)
            gpu_order.append(n)

    # 构造映射：old GPU id -> 新 id
    mapping = {old: i for i, old in enumerate(gpu_order)}
    # print("GPU rename mapping:", mapping)

    # 重命名节点（只改 GPU）
    ring2 = nx.relabel_nodes(ring, mapping, copy=True)

    # 在 graph 元信息中更新 edge_sequence / node_sequence
    def upd_list(lst):
        return [mapping.get(x, x) for x in lst]

    g = ring2.graph
    g["gpu_order"] = [mapping[n] for n in gpu_order]
    if "edge_sequence" in g:
        g["edge_sequence"] = [(mapping.get(u, u), mapping.get(v, v)) for u, v in g["edge_sequence"]]
    if "node_sequence" in g:
        g["node_sequence"] = [mapping.get(n, n) for n in g["node_sequence"]]

    return ring2

import re
import networkx as nx

def write_simai_topo_half_pd_with_exline(
    ring: nx.DiGraph,
    path: str = "My_topo",
    *,
    gpu_per_server: int = 1,
    gpu_type: str = "H100",
    bandwidth_key: str = "bandwidth",        # 若无则兜底用 'capacity'
    prop_key: str = "propagation_delay",     # 单位=秒（原 ring 中）
    n_extra_switches: int = 10,              # 兼容旧签名；本实现不使用
    extra_link_bw: str = "400Gbps",          # 兼容旧签名；本实现不使用
    extra_link_latency: str = "0.0005ms",    # 兼容旧签名；本实现不使用
):
    # ---------- helpers ----------
    def is_gpu(n):
        return str(ring.nodes[n].get("type", "")).upper() == "GPU"

    def to_gbps(val):
        if val is None:
            raise ValueError("边缺少带宽字段（bandwidth/capacity）")
        s = str(val).strip()
        return s if (re.search(r"[A-Za-z]", s) and s.lower().endswith("gbps")) else (s if re.search(r"[A-Za-z]", s) else f"{s}Gbps")

    def sec_to_ms_float(x):
        s = str(x).strip()
        m = re.match(r"^\s*([0-9.]+)\s*ms\s*$", s, flags=re.I)
        if m: return float(m.group(1))
        m = re.match(r"^\s*([0-9.]+)\s*s\s*$", s, flags=re.I)
        if m: return float(m.group(1)) * 1000.0
        return float(s) * 1000.0  # 纯数字按秒

    def ms_str(x): return f"{x}ms"

    # ---------- 节点与编号映射 ----------
    gpu_nodes = [n for n in ring.nodes if is_gpu(n)]
    non_gpu_orig = [n for n in ring.nodes if not is_gpu(n)]

    G = len(gpu_nodes)
    out_id = {}

    # GPUs 放最前（仅为了编号稳定）
    for i, g in enumerate(gpu_nodes):
        out_id[g] = i

    # 每 GPU 一颗 NVS（紧随其后）
    nvs_base = G
    gpu2nvs = {g: (nvs_base + i) for i, g in enumerate(gpu_nodes)}

    # 原非GPU节点
    non_gpu_base = G + len(gpu_nodes)
    for j, n in enumerate(non_gpu_orig):
        out_id[n] = non_gpu_base + j

    # 为每条“有向 GPU→GPU”边分配 1 个中间 switch（MID）
    gpu_gpu_edges = [(u, v, ed) for u, v, ed in ring.edges(data=True) if is_gpu(u) and is_gpu(v)]
    mid_base = non_gpu_base + len(non_gpu_orig)
    edge2mid = {(u, v): (mid_base + k) for k, (u, v, _) in enumerate(gpu_gpu_edges)}

    # ---------- 生成导出边（确保 GPU 只连自家 NVS） ----------
    export_edges = []

    # === 为每个 GPU 统计所有下游/上游链路的平均带宽（Gbps） ===
    def to_gbps_num(val):
        if val is None:
            raise ValueError("边缺少带宽字段（bandwidth/capacity）")
        s = str(val).strip()
        # 提取数字部分
        m = re.search(r"([0-9.]+)", s)
        if not m:
            raise ValueError(f"无法解析带宽值：{val}")
        num = float(m.group(1))
        unit = s.lower()
        if "tbps" in unit:
            num *= 1000.0
        elif "mbps" in unit:
            num /= 1000.0
        # 否则默认按 Gbps
        return num

    def gbps_str(x):
        return f"{int(x)}Gbps" if float(x).is_integer() else f"{x}Gbps"

    out_bw_vals = {}  # GPU -> [下游链路Gbps数值]
    in_bw_vals  = {}  # GPU -> [上游链路Gbps数值]

    for u, v, ed in ring.edges(data=True):
        bw_val = ed.get(bandwidth_key, ed.get("capacity", None))
        bw_num = to_gbps_num(bw_val)
        if is_gpu(u):
            out_bw_vals.setdefault(u, []).append(bw_num)
        if is_gpu(v):
            in_bw_vals.setdefault(v, []).append(bw_num)

    def avg_or_default(vals, default=400.0):
        return (sum(vals) / len(vals)) if vals else default

    # 1) GPU↔NVS：使用平均带宽，传播时延固定 0ms（双向）
    for g in gpu_nodes:
        nvs = gpu2nvs[g]
        bw_g2nvs = gbps_str(avg_or_default(out_bw_vals.get(g)))  # GPU → NVS
        bw_nvs2g = gbps_str(avg_or_default(in_bw_vals.get(g)))   # NVS → GPU
        export_edges.append((out_id[g], nvs, bw_g2nvs, "0ms", "0"))
        export_edges.append((nvs, out_id[g], bw_nvs2g, "0ms", "0"))

    # 2) 改写其余涉及 GPU 的边；非GPU↔非GPU 原样保留
    for u, v, ed in ring.edges(data=True):
        bw = to_gbps(ed.get(bandwidth_key, ed.get("capacity", None)))
        if prop_key not in ed:
            raise ValueError(f"边 {u}->{v} 缺少 '{prop_key}'（单位=秒）")
        pd_ms = sec_to_ms_float(ed[prop_key])

        u_is_gpu, v_is_gpu = is_gpu(u), is_gpu(v)

        if u_is_gpu and v_is_gpu:
            # 原 GPU→GPU ： NVS(u) → MID(u,v) → NVS(v)
            mid = edge2mid[(u, v)]
            half = ms_str(pd_ms / 2.0)
            nu, nv = gpu2nvs[u], gpu2nvs[v]
            export_edges.append((nu, mid, bw, half, "0"))
            export_edges.append((mid, nv, bw, half, "0"))

        elif u_is_gpu and not v_is_gpu:
            # 原 GPU→非GPU ：改成 NVS(u) → 非GPU
            nu, mv = gpu2nvs[u], out_id[v]
            export_edges.append((nu, mv, bw, ms_str(pd_ms), "0"))

        elif not u_is_gpu and v_is_gpu:
            # 原 非GPU→GPU ：改成 非GPU → NVS(v)
            mu, nv = out_id[u], gpu2nvs[v]
            export_edges.append((mu, nv, bw, ms_str(pd_ms), "0"))

        else:
            # 非GPU↔非GPU：原样保留
            mu, mv = out_id[u], out_id[v]
            export_edges.append((mu, mv, bw, ms_str(pd_ms), "0"))

    # ---------- header 与 switch 列表 ----------
    nv_switch_num = len(gpu_nodes)                               # 每 GPU 1 个 NVS
    other_switch_num = len(non_gpu_orig) + len(gpu_gpu_edges)    # 原非GPU + 每条 GPU→GPU 的 MID
    nodes_total = G + nv_switch_num + other_switch_num
    links_total = len(export_edges)

    # 第二行：Switch IDs（顺序：NVS → 原非GPU → MIDs）
    switch_id_line = list(range(nvs_base, nvs_base + nv_switch_num)) + \
                     [out_id[n] for n in non_gpu_orig] + \
                     [edge2mid[(u, v)] for (u, v, _) in gpu_gpu_edges]

    # ---------- 写文件 ----------
    with open(path, "w") as f:
        f.write(f"{nodes_total} {gpu_per_server} {nv_switch_num} {other_switch_num} {links_total} {gpu_type}\n")
        f.write(" ".join(map(str, switch_id_line)) + "\n")
        for (a, b, bw_out, lat_out, er) in export_edges:
            f.write(f"{a} {b} {bw_out} {lat_out} {er}\n")