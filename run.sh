#!/bin/bash
#python experiments/train.py -c baselines/DSTRformer/PEMS04.py --gpus '0'
#python experiments/train.py -c baselines/DSTRformer/PEMS08.py --gpus '0'
#python experiments/train.py -c baselines/DSTRformer/PEMS03.py --gpus '0'
#python experiments/train.py -c baselines/DSTRformer/PEMS07.py --gpus '0'

set CUDA_LAUNCH_BLOCKING=1
python experiments/train.py -c baselines/DSTRformer/STREETS-gurnee.py --gpus '0'