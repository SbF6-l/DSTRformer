"""
STREETS_gurnee/json2npz.py
把原始 trafficcounts 的 json → data.npz
生成文件：datasets/raw_data/STREETS_gurnee/STREETS_gurnee.npz
"""
import os, json, numpy as np, pandas as pd

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

COMMUNITY   = "gurnee"
ROOT_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
RAW_DIR     = os.path.join(ROOT_DIR, "datasets/STREETS")
GRAPH_FILE  = os.path.join(RAW_DIR, "graphs", COMMUNITY, f"{COMMUNITY}-graph.json")
OUT_DIR     = os.path.join(ROOT_DIR, "datasets/raw_data", f"STREETS_{COMMUNITY}")
ensure_dir(OUT_DIR)



# 1. 相机顺序
with open(GRAPH_FILE) as f:
    sensor_dict = json.load(f)["sensor-dictionary"]
sensor_ids = [info[-1] for info in sensor_dict.values()]   # 最后一项就是描述
sensor_ids = sorted(set(sensor_ids))   # 去重并保持一致顺序
N = len(sensor_ids)

print("sensor_ids (graph) =", sensor_ids[:5], "...")
print("traffic 2018 文件数 =", len(os.listdir(os.path.join(RAW_DIR, "trafficcounts", "2018"))))
print("traffic 2018 第一条 key =", list(json.load(open(os.path.join(RAW_DIR, "trafficcounts", "2018", os.listdir(os.path.join(RAW_DIR, "trafficcounts", "2018"))[0]))).keys())[:5])
# 2. 读取 2018+2019
rows = []
for year in ["2018", "2019"]:
    year_dir = os.path.join(RAW_DIR, "trafficcounts", year)
    for fn in sorted(os.listdir(year_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(year_dir, fn)) as f:
            day_data = json.load(f)
        date_str = fn.replace("-trafficcounts.json", "")
        for cam in sensor_ids:
            for rec in day_data.get(cam, []):
                in_num, out_num, hh, mm = rec
                ts = pd.Timestamp(f"{date_str} {hh:02d}:{mm:02d}")
                rows.append([ts, cam, in_num, out_num])

df = pd.DataFrame(rows, columns=["datetime", "camera", "in", "out"])
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.set_index("datetime")

# 3. 5 min 重采样 → (T, N, 2)
# 行=时间戳，列=相机+in/out
df = df.pivot_table(index="datetime",
                    columns=["camera"],
                    values=["in", "out"],
                    aggfunc="sum",
                    fill_value=0)
# 扁平列名：camera_in / camera_out
df.columns = [f"{cam}_{io}" for io, cam in df.columns]
# 重新采样 10 min，补齐缺失行
df = df.resample("10min").sum().fillna(0)


data = df.values.reshape(len(df), N, 2).astype(np.float32)

# 4. 保存
npz_path = os.path.join(OUT_DIR, f"STREETS_{COMMUNITY}.npz")
np.savez(npz_path, data=data)
print(f"Saved {npz_path} shape={data.shape}")

