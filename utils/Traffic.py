import numpy as np

def get_bandwidth_trace(method="uniform", steps=1, capacity=100):
    R_max = capacity  # Gbit/s
    if method == "uniform":
        A = np.random.uniform(0.1, 1.0, steps)
    elif method == "gaussian":
        # 更大方差 + clip 限制范围 0.1 - 1.0
        A = np.clip(np.random.normal(0.5, 0.5, steps), 0.1, 1.0)
    elif method == "onoff":
        A = []
        state = 1
        for _ in range(steps):
            if state == 1:
                A.append(1.0)
                state = 0 if np.random.rand() < 0.1 else 1
            else:
                A.append(0.1)
                state = 1 if np.random.rand() < 0.3 else 0
        A = np.array(A)
    elif method == "random":
        # 离散随机取值：25, 50, 75, ..., 2*capacity
        choices = np.arange(25, 2 * capacity + 1, 25)
        A = np.random.choice(choices, size=steps)
        return float(A[0])  # 单位 Gbit/s，直接返回绝对值
    else:
        raise ValueError(f"Unknown method: {method}")

    return R_max * A[0]  # Gbit/s at each step

# 示例
# print(get_bandwidth_trace(method="gaussian", steps=1, capacity=400))