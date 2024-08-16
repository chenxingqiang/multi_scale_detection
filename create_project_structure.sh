#!/bin/bash

# 创建主目录
mkdir -p multi_scale_detection

# 创建数据集及处理脚本目录
mkdir -p multi_scale_detection/data/ICDAR2015
mkdir -p multi_scale_detection/data/TotalText
touch multi_scale_detection/data/data_loader.py

# 创建模型架构目录
mkdir -p multi_scale_detection/models
touch multi_scale_detection/models/dbnet.py
touch multi_scale_detection/models/titdet.py
touch multi_scale_detection/models/multi_scale_fusion.py

# 创建自定义损失函数目录
mkdir -p multi_scale_detection/loss
touch multi_scale_detection/loss/multi_scale_loss.py

# 创建训练与验证脚本目录
mkdir -p multi_scale_detection/train
touch multi_scale_detection/train/train.py
touch multi_scale_detection/train/validate.py

# 创建辅助工具目录
mkdir -p multi_scale_detection/utils
touch multi_scale_detection/utils/metrics.py
touch multi_scale_detection/utils/visualization.py

# 创建主入口文件
touch multi_scale_detection/main.py

echo "Directory structure created successfully."
