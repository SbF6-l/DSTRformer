"""
DSTRformer/scripts/data_preparation/STREETS_gurnee/generate_training_data.py
"""
import os
import sys
import shutil
import pickle
import argparse
import numpy as np

# 让 basicts 可被 import（保持与原脚本一致）
sys.path.append(os.path.abspath(__file__ + "/../../../.."))
from basicts.data.transform import standard_transform


def generate_data(args: argparse.Namespace):
    target_channel  = args.target_channel
    future_seq_len  = args.future_seq_len
    history_seq_len = args.history_seq_len
    add_time_of_day = args.tod
    add_day_of_week = args.dow
    output_dir      = args.output_dir
    train_ratio     = args.train_ratio
    valid_ratio     = args.valid_ratio
    data_file_path  = args.data_file_path
    steps_per_day   = args.steps_per_day
    norm_each_channel = args.norm_each_channel
    if_rescale = not norm_each_channel

    # 读取数据
    data = np.load(data_file_path)["data"]          # shape = (T, N, 2)
    data = data[..., target_channel]                # 选通道
    l, n, f = data.shape

    # 划分窗口
    num_samples = l - (history_seq_len + future_seq_len) + 1
    train_num = round(num_samples * train_ratio)
    valid_num = round(num_samples * valid_ratio)
    test_num  = num_samples - train_num - valid_num

    index_list = [(t - history_seq_len, t, t + future_seq_len)
                  for t in range(history_seq_len, history_seq_len + num_samples)]
    train_index = index_list[:train_num]
    valid_index = index_list[train_num: train_num + valid_num]
    test_index  = index_list[train_num + valid_num:]

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 归一化
    scaler = standard_transform
    data_norm = scaler(data, output_dir, train_index, history_seq_len,
                       future_seq_len, norm_each_channel=norm_each_channel)

    # 时间特征
    feature_list = [data_norm]
    if add_time_of_day:
        tod = (np.arange(l) % steps_per_day) / steps_per_day
        tod = np.tile(tod, [n, 1]).T[..., None]
        feature_list.append(tod)
    if add_day_of_week:
        dow = (np.arange(l) // steps_per_day) % 7 / 7
        dow = np.tile(dow, [n, 1]).T[..., None]
        feature_list.append(dow)
    processed_data = np.concatenate(feature_list, axis=-1)

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir,
                           f"index_in_{history_seq_len}_out_{future_seq_len}_rescale_{if_rescale}.pkl"), "wb") as f:
        pickle.dump({"train": train_index, "valid": valid_index, "test": test_index}, f)

    with open(os.path.join(output_dir,
                           f"data_in_{history_seq_len}_out_{future_seq_len}_rescale_{if_rescale}.pkl"), "wb") as f:
        pickle.dump({"processed_data": processed_data}, f)



if __name__ == "__main__":
    # ===== 调参区 =====
    HISTORY_SEQ_LEN = 12
    FUTURE_SEQ_LEN  = 12
    TRAIN_RATIO     = 0.6
    VALID_RATIO     = 0.2
    TARGET_CHANNEL  = [0, 1]          # 0: inbound, 1: outbound
    STEPS_PER_DAY   = 144             # 10 min 采样，一天 144 步
    TOD             = True
    DOW             = True
    DATASET_NAME    = "STREETS_gurnee"
    # ==================

    OUTPUT_DIR      = f"datasets/{DATASET_NAME}"
    DATA_FILE_PATH  = f"datasets/raw_data/{DATASET_NAME}/{DATASET_NAME}.npz"   # 你稍后生成
    GRAPH_FILE_PATH = f"datasets/{DATASET_NAME}/adj_mx.pkl"             # 由 generate_adj_mx.py 提供

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",      type=str, default=OUTPUT_DIR)
    parser.add_argument("--data_file_path",  type=str, default=DATA_FILE_PATH)
    parser.add_argument("--graph_file_path", type=str, default=GRAPH_FILE_PATH)
    parser.add_argument("--history_seq_len", type=int, default=HISTORY_SEQ_LEN)
    parser.add_argument("--future_seq_len",  type=int, default=FUTURE_SEQ_LEN)
    parser.add_argument("--steps_per_day",   type=int, default=STEPS_PER_DAY)
    parser.add_argument("--tod",             type=bool, default=TOD)
    parser.add_argument("--dow",             type=bool, default=DOW)
    parser.add_argument("--target_channel",  type=int, nargs="+", default=TARGET_CHANNEL)
    parser.add_argument("--train_ratio",     type=float, default=TRAIN_RATIO)
    parser.add_argument("--valid_ratio",     type=float, default=VALID_RATIO)
    parser.add_argument("--norm_each_channel", action="store_true")
    args = parser.parse_args()

    print("-" * 70)
    for k, v in sorted(vars(args).items()):
        print(f"|{k:>21} = {v}")
    print("-" * 70)

    # 先检查数据够不够
    data = np.load(args.data_file_path)["data"]
    l = data.shape[0]
    need = args.history_seq_len + args.future_seq_len
    min_samples = int((l - need + 1) * (args.train_ratio + args.valid_ratio) + 1)
    if l < need:
        raise ValueError(f"数据行数 {l} < 窗口需要 {need}，无法切分！")
    if (l - need + 1) < 3:
        raise ValueError("可切样本不足 3 个，请补充数据或减小窗口/比例！")
    
    # 一次跑两种归一化
    args.norm_each_channel = True
    generate_data(args)
    args.norm_each_channel = False
    generate_data(args)