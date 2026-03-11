# Deep Hashing Model Training Scripts

本目录包含用于训练深度哈希检索模型的脚本，支持多种主流哈希方法和数据集。

## 支持的哈希方法

| 方法 | 描述 |
|------|------|
| **DPSH** | Deep Pairwise Supervised Hashing |
| **HashNet** | 基于连续哈希的深度哈希网络 |
| **CSQ** | Centralized Supervised Quantization |

## 支持的数据集

- **MIRFlickr-25K**: 25,000 张图像，24 个标签
- **NUS-WIDE**: 81 个标签，评估时使用 Top-21 标签
- **MS-COCO**: 80 个对象类别
- **FLICKR-25K**: 25,000 张图像，多标签设置

## 支持的哈希码长度

- 16-bit
- 32-bit
- 64-bit

## 训练脚本

### MIRFlickr-25K 数据集

```bash
# DPSH
python scripts/train/train_dpsh_mirflickr_16bit.py --gpu 0
python scripts/train/train_dpsh_mirflickr_32bit.py --gpu 0
python scripts/train/train_dpsh_mirflickr_64bit.py --gpu 0

# HashNet
python scripts/train/train_hashnet_mirflickr_16bit.py --gpu 0
python scripts/train/train_hashnet_mirflickr_32bit.py --gpu 0
python scripts/train/train_hashnet_mirflickr_64bit.py --gpu 0

# CSQ
python scripts/train/train_csq_mirflickr_16bit.py --gpu 0
python scripts/train/train_csq_mirflickr_32bit.py --gpu 0
python scripts/train/train_csq_mirflickr_64bit.py --gpu 0
```

### NUS-WIDE 数据集

```bash
# DPSH
python scripts/train/train_dpsh_nuswide_16bit.py --gpu 0
python scripts/train/train_dpsh_nuswide_32bit.py --gpu 0
python scripts/train/train_dpsh_nuswide_64bit.py --gpu 0

# HashNet
python scripts/train/train_hashnet_nuswide_16bit.py --gpu 0
python scripts/train/train_hashnet_nuswide_32bit.py --gpu 0
python scripts/train/train_hashnet_nuswide_64bit.py --gpu 0

# CSQ
python scripts/train/train_csq_nuswide_16bit.py --gpu 0
python scripts/train/train_csq_nuswide_32bit.py --gpu 0
python scripts/train/train_csq_nuswide_64bit.py --gpu 0
```

### MS-COCO 数据集

```bash
# DPSH
python scripts/train/train_dpsh_coco_16bit.py --gpu 0
python scripts/train/train_dpsh_coco_32bit.py --gpu 0
python scripts/train/train_dpsh_coco_64bit.py --gpu 0

# HashNet
python scripts/train/train_hashnet_coco_16bit.py --gpu 0
python scripts/train/train_hashnet_coco_32bit.py --gpu 0
python scripts/train/train_hashnet_coco_64bit.py --gpu 0

# CSQ
python scripts/train/train_csq_coco_16bit.py --gpu 0
python scripts/train/train_csq_coco_32bit.py --gpu 0
python scripts/train/train_csq_coco_64bit.py --gpu 0
```

### FLICKR-25K 数据集

```bash
python scripts/train/train_dpsh_flickr.py --gpu 0
```

## 通用参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gpu` | GPU 设备 ID | 0 |
| `--resume` | 从指定 epoch 恢复训练 | None |

## 训练配置

- **Batch Size**: 128 (DPSH/HashNet), 64 (CSQ)
- **Learning Rate**: 1e-4
- **Epochs**: 60
- **Optimizer**: Adam
- **Backbone**: ResNet50 (ImageNet 预训练)

## 评估协议

- **MIRFlickr-25K**: 使用 Top-21 标签 (24标签中选取出现频率最高的21个)，mAP@5000
- **NUS-WIDE**: 使用 Top-21 标签，mAP@5000
- **MS-COCO**: 使用全量 80 标签，mAP@5000

## 依赖安装

```bash
pip install -r scripts/train/requirements.txt
```

或安装项目根目录依赖：

```bash
pip install -r requirements.txt
```

## 训练输出

训练完成后，模型和日志保存在以下目录结构：

```
results/
├── MIRFlickr25K/
│   ├── DPSH/ResNet50/{BIT}bits/
│   │   ├── checkpoints/
│   │   │   ├── model_best.pth
│   │   │   ├── model_final.pth
│   │   │   └── checkpoint_epoch_*.pth
│   │   └── train_log.json
│   ├── HashNet/ResNet50/{BIT}bits/
│   └── CSQ/ResNet50/{BIT}bits/
├── NUSWIDE_Top21/
│   └── ...
└── COCO/
    └── ...
```

## 数据集准备

确保数据集按照以下结构组织：

```
data/
├── mirflickr25k/
│   ├── mirflickr/
│   │   ├── im1.jpg
│   │   ├── im2.jpg
│   │   └── ...
│   ├── annotations/
│   │   ├── animals.txt
│   │   ├── baby.txt
│   │   └── ...
│   └── split_mirflickr_25k_standard.json
├── NUSWIDE/
│   └── ...
└── coco/
    └── ...
```

## 注意事项

1. 所有脚本使用相同的随机种子 (42) 以确保可复现性
2. 训练过程中每 5 个 epoch 进行一次 mAP 评估
3. 自动保存最佳模型 (best mAP)
4. 支持从中断处恢复训练 (`--resume` 参数)
