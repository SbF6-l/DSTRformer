import os
import json
import pickle
import numpy as np

# ========== 更改社区 ==========
COMMUNITY = "gurnee"
# =============================

ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
GRAPH_FILE = os.path.join(ROOT_DIR, "datasets/STREETS/graphs", COMMUNITY, f"{COMMUNITY}-graph.json")
OUT_DIR    = os.path.join(ROOT_DIR, "datasets/raw_data", f"STREETS_{COMMUNITY}")
os.makedirs(OUT_DIR, exist_ok=True)

# 读入 JSON 中的邻接矩阵
with open(GRAPH_FILE, "r") as f:
    graph = json.load(f)
adj_mx = np.array(graph["adjacency-matrix"], dtype=np.float32)

# 可选：加自环
ADD_SELF_LOOP = False
if ADD_SELF_LOOP:
    adj_mx += np.eye(adj_mx.shape[0])

NUM_NODES = 320
if adj_mx.shape[0] < NUM_NODES:
    pad = NUM_NODES - adj_mx.shape[0]
    adj_mx = np.pad(adj_mx, ((0, pad), (0, pad)), mode='constant', constant_values=0)

# 保存邻接矩阵
adj_file = os.path.join(OUT_DIR, f"adj_mx.pkl")
with open(adj_file, "wb") as f:
    pickle.dump(adj_mx, f)

# 没有距离矩阵，保存 None
dist_file = os.path.join(OUT_DIR, f"adj_STREETS_{COMMUNITY}_distance.pkl")
with open(dist_file, "wb") as f:
    pickle.dump(None, f)

print(f"✅ adjacency matrix saved to {adj_file}, shape = {adj_mx.shape}")
print(f"✅ distance matrix saved to {dist_file} (set to None)")
