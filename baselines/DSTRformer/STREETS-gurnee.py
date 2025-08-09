import os
import sys
import torch
from easydict import EasyDict
from basicts.runners import SimpleTimeSeriesForecastingRunner
from basicts.data import TimeSeriesForecastingDataset
from basicts.losses import masked_mae
from basicts.utils import load_adj

from .arch import DSTRformer

CFG = EasyDict()

# ================= general ================= #
CFG.DESCRIPTION = "DSTRformer config for STREETS_gurnee"
CFG.RUNNER = SimpleTimeSeriesForecastingRunner
CFG.DATASET_CLS = TimeSeriesForecastingDataset
CFG.DATASET_NAME = "STREETS_gurnee"
CFG.DATASET_TYPE = "Traffic flow"
CFG.DATASET_INPUT_LEN = 12
CFG.DATASET_OUTPUT_LEN = 12
CFG.GPU_NUM = 1
CFG.NULL_VAL = 0.0

# ================= environment ================= #
CFG.ENV = EasyDict()
CFG.ENV.SEED = 1
CFG.ENV.CUDNN = EasyDict()
CFG.ENV.CUDNN.ENABLED = True

# ================= model ================= #
CFG.MODEL = EasyDict()
CFG.MODEL.ARCH = DSTRformer
CFG.MODEL.NAME = "STREETS_gurnee_DSTRformer"

# 加载邻接矩阵
adj_mx, _ = load_adj("datasets/STREETS_gurnee/adj_mx.pkl", "doubletransition")
CFG.MODEL.PARAM = {
    "num_nodes": 320,                              # 摄像头数，要与你的实际数据一致
    "adj_mx": [torch.tensor(i) for i in adj_mx],
    "in_steps": 12,
    "out_steps": 12,
    "steps_per_day": 144,                          # 10分钟采样，一天144步
    "input_dim": 4,                                # in, out, TOD/DOW
    "output_dim": 1,
    "input_embedding_dim": 24,
    "tod_embedding_dim": 24,
    "ts_embedding_dim": 28,
    "dow_embedding_dim": 24,
    "time_embedding_dim": 0,
    "adaptive_embedding_dim": 100,
    "node_dim": 64,
    "feed_forward_dim": 256,
    "out_feed_forward_dim": 256,
    "num_heads": 4,
    "num_layers": 2,
    "num_layers_m": 1,
    "dropout": 0.1,
    "mlp_num_layers": 2,
    "use_mixed_proj": True
}
CFG.MODEL.FORWARD_FEATURES = [0, 1, 2, 3]
CFG.MODEL.TARGET_FEATURES = [0]

# ================= optim ================= #
CFG.TRAIN = EasyDict()
CFG.TRAIN.LOSS = masked_mae
CFG.TRAIN.OPTIM = EasyDict()
CFG.TRAIN.OPTIM.TYPE = "Adam"
CFG.TRAIN.OPTIM.PARAM = {
    "lr": 0.001,
    "weight_decay": 0.0015,
}
CFG.TRAIN.LR_SCHEDULER = EasyDict()
CFG.TRAIN.LR_SCHEDULER.TYPE = "MultiStepLR"
CFG.TRAIN.LR_SCHEDULER.PARAM = {
    "milestones": [25, 45, 65],
    "gamma": 0.1
}

# ================= train ================= #
CFG.TRAIN.NUM_EPOCHS = 80
CFG.TRAIN.CKPT_SAVE_DIR = os.path.join("checkpoints", CFG.MODEL.NAME + "_" + str(CFG.TRAIN.NUM_EPOCHS))
CFG.TRAIN.DATA = EasyDict()
CFG.TRAIN.DATA.DIR = "datasets/STREETS_gurnee"
CFG.TRAIN.DATA.BATCH_SIZE = 16
CFG.TRAIN.DATA.PREFETCH = False
CFG.TRAIN.DATA.SHUFFLE = True
CFG.TRAIN.DATA.NUM_WORKERS = 2
CFG.TRAIN.DATA.PIN_MEMORY = False

# ================= validate ================= #
CFG.VAL = EasyDict()
CFG.VAL.INTERVAL = 1
CFG.VAL.DATA = EasyDict()
CFG.VAL.DATA.DIR = "datasets/STREETS_gurnee"
CFG.VAL.DATA.BATCH_SIZE = 16
CFG.VAL.DATA.PREFETCH = False
CFG.VAL.DATA.SHUFFLE = False
CFG.VAL.DATA.NUM_WORKERS = 2
CFG.VAL.DATA.PIN_MEMORY = False

# ================= test ================= #
CFG.TEST = EasyDict()
CFG.TEST.INTERVAL = 1
CFG.TEST.DATA = EasyDict()
CFG.TEST.DATA.DIR = "datasets/STREETS_gurnee"
CFG.TEST.DATA.BATCH_SIZE = 16
CFG.TEST.DATA.PREFETCH = False
CFG.TEST.DATA.SHUFFLE = False
CFG.TEST.DATA.NUM_WORKERS = 2
CFG.TEST.DATA.PIN_MEMORY = False
