# KV 缓存
cache = {}

def add_to_cache(step, key, value):
    """把某一步的注意力计算结果放入缓存"""
    cache[step] = (key, value)

def get_from_cache(step):
    """取出缓存内容"""
    return cache.get(step, None)


# 模拟推理过程
tokens = ["你", "好", "世", "界"]

for step, token in enumerate(tokens):
    key, value = f"key_{token}", f"value_{token}"

    # 把计算结果加入缓存
    add_to_cache(step, key, value)

    print(f"第 {step} 步生成: {token}, 缓存中存了 -> {cache[step]}")



#请求队列
import heapq

request_q = []

# 模拟一些请求 (priority, request内容)
requests = [
    (2, "生成摘要 A"),
    (1, "生成翻译 B"),
    (3, "回答问题 C")
]

for priority,request in requests:
    heapq.heappush(request_q,(priority,request))

while request_q:
    priority,request= heapq.heappop(request_q)


#动态 batching（简单版）

"""
场景设定
有一堆推理请求 requests
批量大小设为 4（也可以是别的数字）
每当积累到 4 个请求就一起推理（调用 run_inference）
剩余不足 4 个的请求最后也要处理
"""

# 模拟请求列表
requests = ["req1", "req2", "req3", "req4", "req5", "req6", "req7"]

# 批量存储
batch = []

# 模拟推理函数
def run_inference(batch):
    print(f"推理批次: {batch}")

for req in requests:
    batch.append(req)
    if len(batch)==4:
        run_inference(batch)
        batch=[]
if batch:
    run_inference(batch)

"""
量化推理（Quantization）模拟

场景：把浮点数权重转换成低精度（如 int8）来加速推理

练习点：

模拟权重压缩

比较原始 vs 量化后的推理结果差异
"""

import numpy as np

weights = np.array([0.1, -0.5, 0.8, 1.2])
int8_weights = np.round(weights * 127).astype(np.int8)
print("量化后的权重:", int8_weights)

"""
剪枝（Pruning）模拟

场景：把模型中小权重置零，减少计算量

练习点：

简单判断哪些权重小于阈值就置零

模拟稀疏矩阵乘法
"""

weights = np.array([0.1, -0.05, 0.8, 0.02])
pruned = np.where(np.abs(weights) < 0.05, 0, weights)
print("剪枝后的权重:", pruned)

"""
缓存命中率统计

场景：模拟 KV Cache 时，统计重复 token 的缓存命中率

练习点：判断历史 token 是否已经缓存，统计命中次数
"""
cache = {}
tokens = ["你", "好", "你", "世", "界", "好"]
hit, miss = 0, 0

for step, token in enumerate(tokens):
    if token in cache:
        hit += 1
    else:
        miss += 1
        cache[token] = f"value_{token}"

print(f"缓存命中 {hit} 次，未命中 {miss} 次")
