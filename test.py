# -*- coding: utf-8 -*-
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from baseline import baseline
from Fast import fast
from Slow import slow
from FastSlow import fastslow

def run_one(algo_name, packet):
    packet_size = packet / 1024.0
    if algo_name == "baseline":
        return baseline(packet_size)
    elif algo_name == "fast":
        return fast(packet_size)
    elif algo_name == "slow":
        return slow(packet_size)
    elif algo_name == "fastslow":
        return fastslow(packet_size)

if __name__ == "__main__":
    packet_list = [8, 16, 32, 64]
    algos = ["baseline", "fast", "slow", "fastslow"]

    num_processes = 20  # <<< 改进程数量这里改

    results_dict = {}

    with ProcessPoolExecutor(max_workers=num_processes) as pool:
        futures = {}
        for i, packet in enumerate(packet_list):
            for j, algo in enumerate(algos):
                fut = pool.submit(run_one, algo, packet)
                futures[fut] = (packet, algo, i, j)

        for fut in as_completed(futures):
            packet, algo, i, j = futures[fut]
            try:
                result = fut.result()
            except Exception as e:
                result = f"ERROR: {e}"
            results_dict[(packet, algo)] = result

    # 重新组装成二维列表，确保顺序完全一致
    collective_time = [
        [results_dict[(pkt, algo)] for algo in algos]
        for pkt in packet_list
    ]

    for pkt, row in zip(packet_list, collective_time):
        print(f"packet={pkt}KB -> {row}")

    print("\nFinal collective_time:", collective_time)