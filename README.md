<div align="center">
  <h1>CaptionLens</h1>
  <p><strong>Image Semantic Captioning with Self-Attention and Cross-Attention</strong></p>
  <p>融合自注意力与交叉注意力机制的图像语义描述生成系统</p>
</div>

## 项目简介

CaptionLens 是一个基于深度学习的图像语义描述生成系统。项目以 ResNet50 作为 CNN 图像编码器，从输入图片中提取区域视觉特征；再使用 Transformer Decoder 作为文本解码器，通过 masked self-attention 建模已生成词语之间的上下文关系，并通过 cross-attention 对齐文本 token 与图像区域特征，最终逐词生成自然语言 caption。

项目覆盖了图像描述任务的完整实验流程，包括 MSCOCO 2017 数据下载、词表构建、模型训练、模型推理、自动评估和 Streamlit 前端展示。当前前端已经优化为更适合课程项目、科研项目展示和保研面试讲解的 AI Demo 页面。

项目仓库：[cpn-cyber/CaptionLens](https://github.com/cpn-cyber/CaptionLens)

## 项目工作流

下图展示了 CaptionLens 从图像输入、预处理、ResNet50 视觉编码、Transformer 解码、逐词生成到最终 caption 输出的完整流程，也补充展示了 MSCOCO 2017、词表构建、训练、权重保存和 Streamlit 前端演示之间的关系。

![CaptionLens workflow](screenshots/workflow.png)

## 功能亮点

- **ResNet50 Encoder**：使用预训练 ResNet50 提取图像高层语义特征。
- **14 x 14 Visual Tokens**：将图像转换为 196 个区域视觉 token，便于 Transformer 进行图文对齐。
- **Transformer Decoder**：使用自注意力和交叉注意力完成文本生成。
- **MSCOCO 2017 Training**：支持下载官方 train2017 / val2017 数据并完成训练与验证。
- **BLEU / CIDEr Evaluation**：在独立验证集上对比 Greedy 与 Beam Search 的生成质量。
- **Streamlit Demo**：支持上传 jpg、png、jpeg 图片，并实时展示生成描述。
- **现代化前端展示**：页面包含项目标题、技术标签、上传区域、结果卡片、模型流程说明和项目信息侧边栏。

## Demo 展示

以下截图均为本地重新启动 Streamlit 后，上传测试图片得到的真实推理结果。

### 示例一：夜间交通场景

模型输出：

```text
a view of a traffic light at night
```

![CaptionLens traffic night demo](screenshots/captionlens_traffic_night_demo_cropped.png)

### 示例二：黑白老照片

模型输出：

```text
a man is sitting on a motorcycle in the road
```

这张样例是一张黑白老照片，可以体现模型不仅能处理普通彩色图像，也能对低色彩信息、历史照片风格的图像进行基础语义分析。

![CaptionLens black-and-white motorcycle demo](screenshots/captionlens_bw_motorcycle_demo_cropped.png)

## 系统流程

```text
Input Image
  -> Resize / Normalize
  -> ResNet50 CNN Encoder
  -> 14 x 14 regional visual tokens
  -> Transformer Decoder
     -> Masked Self-Attention
     -> Cross-Attention over image features
  -> Word-by-word Caption Generation
  -> Final Caption
```

页面下方的 “How it works” 区域也对应展示了系统的四个关键步骤：

| 模块 | 作用 |
| --- | --- |
| CNN Encoder | 从输入图像中提取区域视觉特征 |
| Visual Tokens | 将图像表示为 14 x 14 个视觉区域 token |
| Self-Attention | 建模生成词之间的上下文依赖 |
| Cross-Attention | 将文本 token 与图像区域特征对齐 |

## 模型原理

### 1. 图像编码器

`models/encoder.py` 中的 `CNNEncoder` 使用 `torchvision.models.resnet50(pretrained=True)` 加载预训练 ResNet50，并移除最后的分类层。编码器保留卷积特征图，通过 `AdaptiveAvgPool2d((14, 14))` 固定空间尺寸，再使用 `1x1 Conv` 将通道维度压缩到 Transformer 使用的 `embed_size`。

输出形状可以理解为：

```text
(B, 3, H, W)
  -> ResNet50 convolutional features
  -> (B, 2048, 14, 14)
  -> 1x1 Conv
  -> (B, embed_size, 14, 14)
  -> flatten
  -> (B, 196, embed_size)
```

其中 196 个向量就是图像的区域视觉 token。

### 2. Transformer 文本解码器

`models/decoder.py` 中的 `TransformerDecoder` 包含词嵌入、位置编码和多层 Transformer Decoder Layer。解码器在每一步生成时都会使用 causal mask，保证当前位置只能看到已经生成的词，不能提前看到未来词。

每一层 Decoder 主要包含：

- **Masked Self-Attention**：学习 caption 内部词与词之间的关系。
- **Cross-Attention**：根据当前文本状态关注图像区域 token。
- **Feed Forward Network**：进一步进行非线性特征变换。

### 3. 训练目标

训练阶段使用 teacher forcing：给定真实 caption 的前缀，模型预测下一个词。损失函数为 `CrossEntropyLoss`，并忽略 `<PAD>` token 对 loss 的影响。

```text
<START> a man riding a horse
          -> predict next token step by step
```

### 4. 推理方式

当前前端推理函数从 `<START>` 开始自回归生成 caption，并在每一步调用 Transformer Decoder 预测下一个 token。为了减少重复短语和 `<UNK>`，前端推理中加入了 beam search、重复 token 惩罚、no-repeat bigram 约束和 `<UNK>` 惩罚，使输出比最基础的 greedy decoding 更稳定。

## 项目结构

```text
CaptionLens/
├── app.py                         # Streamlit 前端页面与图像上传推理入口
├── train.py                       # 模型训练脚本
├── infer.py                       # 推理函数
├── evaluate.py                    # BLEU-1~4、CIDEr 评估与解码策略对比
├── prepare_splits.py              # 按图像 ID 划分训练集与验证集
├── build_vocab.py                 # 根据 COCO caption 构建词表
├── download_coco2017.py           # 下载官方 MSCOCO 2017 train/val 数据
├── download_coco_subset.py        # 下载小规模 COCO 子集，便于快速调试
├── requirements.txt               # Python 依赖
├── config/
│   └── config.yaml                # 数据、模型和训练超参数
├── models/
│   ├── encoder.py                 # ResNet50 图像编码器
│   ├── decoder.py                 # Transformer 文本解码器
│   └── image_captioning.py        # Encoder + Decoder 总模型
├── utils/
│   ├── dataset.py                 # COCO 图像-caption 数据集封装
│   ├── decoding.py                # Greedy、Beam Search 与生成约束
│   └── tokenizer.py               # 文本清洗、词表、编码与解码
└── screenshots/
    ├── workflow.png
    ├── captionlens_traffic_night_demo_cropped.png
    └── captionlens_bw_motorcycle_demo_cropped.png
```

## 快速开始

### 1. 安装依赖

```powershell
cd C:\Users\13178\Desktop\CaptionLens
pip install -r requirements.txt
```

### 2. 下载 MSCOCO 2017

```powershell
python download_coco2017.py
```

脚本会下载 COCO 2017 captions 标注、train2017 和 val2017 图像。默认路径如下：

```text
data/raw/annotations/captions_train2017.json
data/raw/annotations/captions_val2017.json
data/raw/train2017
data/raw/val2017
```

如果只是本地快速调试，也可以运行小规模子集脚本：

```powershell
python download_coco_subset.py
```

### 3. 构建词表

```powershell
python build_vocab.py
```

输出文件：

```text
data/vocab.pkl
```

词表只根据 train2017 caption 构建，不读取 val2017 文本。

### 4. 训练模型

```powershell
python train.py
```

训练后会保存模型权重：

```text
checkpoints/best_model.pth
```

训练脚本每轮计算 val2017 验证损失，并根据最低验证损失保存最佳权重，同时生成 `checkpoints/training_metadata.json` 记录训练与验证数据来源。

### 5. 评估生成质量

```powershell
python evaluate.py
```

评估脚本会在 val2017 上分别运行 Greedy Search 和 Beam Search，并计算：

- BLEU-1、BLEU-2、BLEU-3、BLEU-4：衡量生成描述与人工参考描述的 n-gram 重合程度。
- CIDEr：使用 TF-IDF 加权的 n-gram 相似度衡量生成描述与多条人工参考描述的一致性。
- Beam vs Greedy：输出每个指标的绝对变化和相对变化百分比。

结果会写入：

```text
evaluation/results.json
```

快速检查代码是否能够运行时，可以只评估少量图片：

```powershell
python evaluate.py --max-images 20
```

少量图片结果仅用于程序检查，正式报告应使用完整 val2017。旧权重如果不是基于当前 train2017 / val2017 配置训练得到，不能用于证明泛化效果；需要依次重新运行 `download_coco2017.py`、`build_vocab.py` 和 `train.py` 后再进行正式评估。

### 6. 启动前端

```powershell
streamlit run app.py
```

如果默认端口被占用，可以指定端口：

```powershell
streamlit run app.py --server.port 8503
```

访问地址：

```text
http://localhost:8501
```

或：

```text
http://localhost:8503
```

## 当前本地训练情况

当前仓库默认不包含预训练权重和词表文件，因此需要先完成数据下载、词表构建和模型训练。训练完成后会生成：

```text
data/vocab.pkl
checkpoints/best_model.pth
```

模型效果与训练轮数、硬件资源和数据规模有关。使用完整 MSCOCO 2017 训练后，可通过 `evaluate.py` 在 val2017 上输出 BLEU 与 CIDEr 等指标，用于量化比较 Greedy Search 与 Beam Search 的生成质量。

## 技术栈

| 模块 | 技术 |
| --- | --- |
| 深度学习框架 | PyTorch |
| 图像编码器 | ResNet50, TorchVision |
| 文本解码器 | Transformer Decoder |
| 注意力机制 | Self-Attention, Cross-Attention |
| 数据集 | MSCOCO 2017 Captions |
| 图像处理 | Pillow, TorchVision Transforms |
| 配置管理 | YAML |
| Web Demo | Streamlit |

## 前端设计

当前 `app.py` 使用 Streamlit + HTML/CSS 实现了一个简洁专业的 AI 项目展示页面，主要包括：

- 顶部项目 Header：展示 CaptionLens、英文副标题和中文说明。
- 技术标签：Transformer、Self-Attention、Cross-Attention、MSCOCO、Streamlit。
- 上传区域：支持 jpg、png、jpeg 图片上传，并展示圆角原图。
- 结果区域：用浅蓝色卡片突出展示生成 caption。
- 流程说明：用四个小卡片解释 CNN Encoder、Visual Tokens、Self-Attention、Cross-Attention。
- 侧边栏信息：展示模型结构、注意力机制、数据集、解码方式和运行设备。
- 指标展示：存在 `evaluation/results.json` 时，侧边栏自动显示 Beam Search 的 BLEU-4 与 CIDEr。

## 局限与改进方向

- 当前模型结构相对轻量，表达能力弱于 BLIP、ViT-GPT2 等大规模预训练视觉语言模型。
- 词表基于简单英文分词，低频词会被映射为 `<UNK>`。
- 当前自动评估包含 BLEU 与 CIDEr，尚未加入 METEOR、ROUGE-L、SPICE 等补充指标。
- 指标结果会受到训练数据规模、训练轮数和词表覆盖率影响，需要结合定性样例分析。
- 可以进一步增加训练轮数、扩大 batch size，并系统调优 beam search 参数。
- 后续可以加入 attention heatmap，可视化模型在生成某个词时关注的图像区域。

## 总结

CaptionLens 展示了一个典型的 CNN Encoder + Transformer Decoder 图像描述系统：先用 CNN 理解图像，再用 Transformer 注意力机制完成图文对齐和自然语言生成。相比只展示单张图片预测结果的简单 demo，本项目包含数据准备、词表构建、模型训练、推理生成和前端展示，适合作为图像语义理解、视觉语言建模和 Transformer 注意力机制的综合实践项目。

## License

This project is licensed under the MIT License. Copyright (c) 2026 cpn-cyber.
