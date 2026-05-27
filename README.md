# Meta Distillation 水印模型训练

这是一个用于水印模型元蒸馏训练的Python程序，支持多种模型架构和数据集。

## 文件说明

- `t2s_v2.py` - 主程序文件
- `config_template.yaml` - YAML配置文件模板
- `run_t2s.sh` - 运行脚本
- `README.md` - 使用说明文档

## 环境要求

- Python 3.6+
- PyTorch
- torchvision
- 其他依赖包（见代码中的import语句）

## 快速开始

### 1. 使用bash脚本运行（推荐）

```bash
# 基本用法
./run_t2s.sh -m res18 -d cifar100 -i 1

# 查看帮助
./run_t2s.sh --help

# 使用自定义参数
./run_t2s.sh -m mobilevit -d tinyimagenet --epochs 5 --alpha 2 --lr-inner 0.0001
```

### 2. 直接使用Python运行

```bash
python t2s_v2.py --model res18 --dataset cifar100 --idx 1 --epochs 3
```

## 配置说明

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model` | str | res18 | 模型选择 (res18, wrn16_4, dense121, googlenet, mobilevit) |
| `--dataset` | str | cifar100 | 数据集 (cifar10, cifar100, tinyimagenet) |
| `--idx` | int | 0 | 实验索引 |
| `--mode` | str | feature | 水印模式 (feature, random_trigegr) |
| `--epochs` | int | 3 | 训练轮数 |
| `--lr_outer` | float | 0.005 | 外部学习率 |
| `--lr_inner` | float | 0.001 | 内部学习率 |
| `--alpha` | int | 20 | 平衡参数 |
| `--device` | str | cuda:0 | 运行设备 |
| `--data_path` | str | /usr/common/datasets/ | 数据集路径 |
| `--image_size` | int | 32 | 图像尺寸 |
| `--inner_batch_size` | int | 100 | 内部批次大小 |
| `--inner_distill_batch_size` | int | 10 | 内部蒸馏批次大小 |
| `--outer_batch_size` | int | 500 | 外部批次大小 |

### YAML配置文件

使用 `config_template.yaml` 作为模板创建自定义配置文件：

```yaml
# 基本设置
experiment:
  idx: 0
  mode: "feature"
  device: "cuda:0"

# 模型和数据集
model:
  name: "res18"
  image_size: 32

dataset:
  name: "cifar100"
  data_path: "/usr/common/datasets/"

# 训练参数
training:
  epochs: 3
  lr_outer: 0.005
  lr_inner: 0.001
  alpha: 20
```

## 使用示例

### 示例1：CIFAR-100上的ResNet-18训练

```bash
./run_t2s.sh -m res18 -d cifar100 -i 1 --epochs 3 --alpha 15
```

### 示例2：TinyImageNet上的MobileViT训练

```bash
./run_t2s.sh -m mobilevit -d tinyimagenet -i 2 --epochs 5 --alpha 2 --lr-inner 0.0001
```

### 示例3：使用配置文件

```bash
# 1. 复制并修改配置文件
cp config_template.yaml my_config.yaml
# 编辑 my_config.yaml

# 2. 使用配置文件运行
./run_t2s.sh -c my_config.yaml
```

### 示例4：干运行（只显示命令不执行）

```bash
./run_t2s.sh -m res18 -d cifar10 --dry-run
```

## 输出文件

程序运行后会在以下目录生成结果：

```
checkpoint/{dataset}/{model}/{idx}/t2s/{mode}/
├── checkpoint.pt          # 训练好的模型
└── experiment_log.json    # 实验日志
```

实验日志包含：
- 实验参数
- 训练时间
- 最终准确率（T-Acc, T-WSR, S-Acc, S-WSR）

## 参数调优建议

### 不同数据集的推荐参数

| 数据集 | alpha | lr_inner | 备注 |
|--------|-------|----------|------|
| CIFAR-10 | 50 | 0.001 | 标准设置 |
| CIFAR-100 | 15 | 0.001 | 平衡参数较小 |
| TinyImageNet | 2 | 0.001 | 平衡参数最小 |

### 不同模型的推荐参数

| 模型 | lr_inner | 备注 |
|------|----------|------|
| ResNet-18 | 0.001 | 标准设置 |
| MobileViT | 0.0001 | 学习率较小 |
| DenseNet-121 | 0.001 | alpha=30 |
| WideResNet | 0.001 | 标准设置 |
| GoogleNet | 0.001 | 标准设置 |

## 故障排除

### 常见问题

1. **CUDA内存不足**
   - 减小批次大小：`--inner_batch_size 50 --outer_batch_size 250`
   - 使用CPU：`--device cpu`

2. **数据集路径错误**
   - 检查 `--data_path` 参数
   - 确保数据集已正确下载

3. **模型文件不存在**
   - 确保教师模型和学生模型已预训练
   - 检查路径：`checkpoint/{dataset}/{model}/{idx}/watermarked/{mode}/checkpoint.pt`
   - 检查路径：`checkpoint/{dataset}/{model}/{idx}/clean/checkpoint.pt`

4. **权限问题**
   - 确保脚本有执行权限：`chmod +x run_t2s.sh`

### 调试模式

使用 `--verbose` 参数获取详细输出：

```bash
./run_t2s.sh -m res18 -d cifar100 --verbose
```

## 注意事项

1. 确保有足够的磁盘空间存储模型和日志文件
2. 训练时间取决于数据集大小、模型复杂度和硬件配置
3. 建议在GPU上运行以获得更好的性能
4. 定期检查实验日志以监控训练进度

## 许可证

请根据项目许可证使用此代码。