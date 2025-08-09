import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

COMMUNITY = "gurnee"
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
RAW_DIR = os.path.join(ROOT_DIR, "datasets/STREETS")
TRAFFIC_DIR = os.path.join(RAW_DIR, "trafficcounts")
OUT_DIR = os.path.join(ROOT_DIR, "datasets/raw_data", f"STREETS_{COMMUNITY}")
ensure_dir(OUT_DIR)

# Step 1: 收集所有摄像头名
camera_names = set()
for year in ["2018", "2019"]:
    year_dir = os.path.join(TRAFFIC_DIR, year)
    for fn in sorted(os.listdir(year_dir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(year_dir, fn)
        with open(path) as f:
            day_data = json.load(f)
        camera_names.update(day_data.keys())

camera_names = sorted(camera_names)
N = len(camera_names)
print(f"有效摄像头数: {N}")
print("摄像头示例:", camera_names[:5])

# Step 2: 提取所有记录
rows = []
for year in ["2018", "2019"]:
    year_dir = os.path.join(TRAFFIC_DIR, year)
    for fn in sorted(os.listdir(year_dir)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(year_dir, fn)
        with open(path) as f:
            day_data = json.load(f)

        for cam, img_dict in day_data.items():
            for img_name, info in img_dict.items():
                try:
                    ts = pd.to_datetime(datetime.strptime(img_name.replace(".jpg", ""), "%Y-%m-%d-%H-%M"))
                    inbound = info.get("inbound", 0)
                    outbound = info.get("outbound", 0)
                    rows.append([ts, cam, inbound, outbound])
                except:
                    continue

# Step 3: 构建 DataFrame
if not rows:
    print("❌ 未找到任何有效数据。")
    exit()

df = pd.DataFrame(rows, columns=["datetime", "camera", "in", "out"])
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.set_index("datetime")

# Step 4: 构造成 (T, N, 2)
df = df.pivot_table(index="datetime",
                    columns=["camera"],
                    values=["in", "out"],
                    aggfunc="sum",
                    fill_value=0)
df.columns = [f"{cam}_{io}" for io, cam in df.columns]
df = df.resample("10min").sum().fillna(0)

# 确保摄像头顺序与原始顺序一致
for cam in camera_names:
    for io in ["in", "out"]:
        col = f"{cam}_{io}"
        if col not in df.columns:
            df[col] = 0

df = df[[f"{cam}_{io}" for cam in camera_names for io in ["in", "out"]]]

data = df.values.reshape(len(df), N, 2).astype(np.float32)

# Step 5: 保存
npz_path = os.path.join(OUT_DIR, f"STREETS_{COMMUNITY}.npz")
np.savez(npz_path, data=data)
print(f"✅ 已保存 {npz_path}")
print(f"数据 shape = {data.shape}, 时间点 = {len(df)}, 摄像头数 = {N}")
