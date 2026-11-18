# RNN-to-Transformer Projects

这个仓库包含两个把已有 RNN 项目升级为 Transformer 的完整工程：

1. `poetry-transformer`：基于 RNN 唐诗生成器的数据和词级预处理，
   使用从零实现的 Transformer 语言模型生成唐诗。
2. `translator-transformer`：基于 RNN 中英翻译器的平行语料、中文 jieba
   词表，以及 Transformer 作业中的 byte-level BPE，构建 Transformer
   编码器-解码器翻译器。

项目中的数据清洗结果、语料、中文分词方式和词表来自以下已经发布的 RNN
仓库：

- [Tang-Poetry-Generator](https://github.com/Twilight-0419/Tang-Poetry-Generator)
- [Chinese-English-Translator](https://github.com/Twilight-0419/Chinese-English-Translator)

Transformer、RoPE、RMSNorm、SwiGLU、缩放点积注意力、AdamW、交叉熵和
学习率调度复用自 NLP 课程 `hw1_bpe_and_lm` 中从零实现的代码。两个项目
新增了适配任务的数据管线、训练入口、检查点格式和命令行推理逻辑。

为避免在新仓库重复保存近百 MB 数据，体积较大的原始语料和可重建预处理
文件仍由上面的两个 RNN 仓库提供。各子项目 README 给出了所需文件的下载
和最简准备方式。

## RNN 与 Transformer 对比

两个项目保留了相同的输入数据和无监督/平行语料目标，但序列建模方式发生了
变化。RNN 通过隐藏状态逐个时间步传递信息，Transformer 则通过注意力在
同一层内直接连接所有 token。

| 对比维度 | RNN / GRU 版本 | Transformer 版本 |
|---|---|---|
| 核心计算 | 时间步递推：`h_t = f(h_{t-1}, x_t)` | 多头自注意力：每个 token 直接读取其他 token |
| 时间维并行 | 较差，必须按顺序计算 | 训练时可在序列维并行 |
| 长距离依赖 | 容易受梯度衰减影响，GRU 只能缓解 | 任意两个位置可直接交互 |
| 位置信息 | 由时间步顺序隐式提供 | 需要显式位置编码，本项目使用 RoPE |
| 计算复杂度 | 通常随序列长度线性增长 | 注意力为 `O(n²)`，长序列成本更高 |
| 归纳偏置 | 强顺序偏置，适合小数据和短句 | 更灵活，但通常需要更多数据 |
| 工程复杂度 | 结构简单、参数机制直观 | 层数、注意力 mask、位置编码和显存管理更复杂 |
| 翻译对齐 | GRU 解码器 + 加性注意力 | 多层自注意力 + 多头跨注意力 |
| 词表问题 | RNN 翻译器使用词级英文词表，存在 `<unk>` | 英文 BPE 可直接处理字节，不产生 `<unk>` |
| 生成方式 | 每步把隐藏状态传给下一步 | Decoder 使用因果 mask 自回归生成 |
| 主要风险 | 长距离信息压缩、重复生成 | 小数据过拟合、显存占用和注意力开销 |

### 具体生成样例对比

下面的翻译样例使用同一批中文输入，分别调用仓库中的 RNN 翻译器和
Transformer 翻译器。两者都使用贪心解码，Transformer 额外对 BPE 解码
结果做了空格、标点和首字母整理。

| 中文输入 | RNN 翻译器 | Transformer 翻译器 |
|---|---|---|
| 娜塔莎，我爱你。 | `natasha , i love you .` | `Natasha, l love you so much.` |
| 你叫什么名字？ | `what 's your name ?` | `What's your name?` |
| 我不知道该怎么办。 | `i don 't know what to do .` | `I don't know what to do.` |
| 今天天气很好。 | `it was good .` | `It's good to be here.` |
| 他在哪里？ | `where 's he ?` | `Where is he?` |

从这组样例可以看到：

- Transformer 在大小写、缩写和标点排版上明显更自然。
- 对“你叫什么名字？”和“我不知道该怎么办。”，两种模型都表达了正确语义。
- “娜塔莎，我爱你。”中 Transformer 输出了 `l` 而不是 `I`，说明它仍会把
  相似字节/字母混淆。
- “今天天气很好。”两者都没有准确表达天气含义，说明样例质量不能代替完整
  测试集评估。

唐诗使用相同起始词“春”各生成一次。RNN 参数为 temperature 0.6、
Top-K 10；Transformer 参数为 temperature 0.8、Top-P 0.9。

**RNN 唐诗生成器：**

```text
春。蛾眉秋不下，何处去更年深。铜缾囊，比珠泪被。
死蛟浮出不复灵，舳舻千里江水清。鼓颦余响数日在，
天吴深入鱼鼈腥。
```

**Transformer 唐诗生成器：**

```text
春柳，粧看香风过。
天际东臯落，萧萧最碧新。
秋风二月色，明月雨中吹。
月照汀洲夜，青山高枕上。
更看暮孤屿，肠断关山多。
```

这组样例中，Transformer 的季节意象更连贯，五言式句子更稳定，也能按规则
持续生成到中文句号。RNN 输出包含更多跳跃意象和生僻字，句间关系较弱。
但两者都仍会生成不自然的古字和个别生硬搭配，说明架构升级不能代替更大的
高质量语料和人工评估。

> 注意：以上是单次采样样例，不是严格的 BLEU、chrF、困惑度或人工盲评结果。
> 两个模型使用相同的原始数据来源，但训练轮数、词表和硬件配置并不完全相同，
> 因此这些例子用于观察行为差异，不用于宣称绝对性能高低。

### 唐诗生成项目中的差异

RNN 版本依靠单个隐藏状态沿诗句向前传递上下文，因此对主题、意象和句式的
长期规划能力有限。Transformer 的因果注意力可以直接回看更长的上下文，
配合 RoPE、Top-K/Top-P 和重复惩罚，能够生成更灵活的长句。

但 Transformer 并不会因为训练更久而必然更好。当前唐诗语料只有约 215 万
token，较大的模型很容易记住训练集，表现为训练 loss 很低而验证 loss
持续上升。因此本项目更适合使用较小模型、权重衰减和基于验证集的提前停止。

### 机器翻译项目中的差异

RNN 翻译器使用 GRU 编码器-解码器和加性注意力，v0.2 约有 2,539 万参数。
本项目默认的 4 层编码器 + 4 层解码器 Transformer 约有 2,364 万参数，
两者处于同一量级，但 Transformer 的多头跨注意力能更直接地建模中英文之间
的词序调整和远距离对应关系。

英文端从 29,754 词的词级词表改为 10,000-token byte-level BPE 后，不再受到
词级 `<unk>` 的限制，也更适合处理缩写、标点和未登录词。代价是 BPE 序列通常
比英文单词序列更长，训练和解码成本相应增加。

## 结论

1. **RNN 仍然适合作为小型、快速、低资源基线。** 它在短句、数据有限、
   推理环境受限或需要简单快速复现时仍然有价值。
2. **Transformer 更适合 GPU 训练、长上下文和复杂的跨序列对齐。**
   对于机器翻译，多编码器/解码器层、跨注意力和 BPE 是明确的升级方向。
3. **Transformer 不是自动胜过 RNN。** 唐诗项目说明，当数据规模有限时，
   更大的 Transformer 更容易过拟合；模型容量、正则化和验证集早停比单纯
   增加训练时间更重要。
4. **最合理的工程选择是保留 RNN 的数据和预处理，按任务选择模型。**
   小数据和低资源推理继续使用 RNN 是合理方案；需要更强表达能力和 GPU
   并行训练时再切换到 Transformer。

## 项目一：Transformer 唐诗生成器

目录：[`poetry-transformer`](poetry-transformer/)

- 输入：起始字或起始短语。
- 数据：RNN 项目清洗后的《全唐诗》语料和 `preprocessing.pt`。
- 模型：Decoder-only Transformer，使用 RoPE、RMSNorm、SwiGLU 和因果注意力。
- 生成：温度、Top-K、Top-P、重复惩罚，以及达到目标长度后等待中文句号。

```bash
cd poetry-transformer
python -m pip install -r requirements.txt
python -m src.prepare_data --overwrite
python -m src.train --device cuda
python -m src.main --start 春 --length 40
```

## 项目二：Transformer 中英翻译器

目录：[`translator-transformer`](translator-transformer/)

- 输入：中文短句。
- 数据：RNN 翻译项目中的 138 万中英平行句对。
- 中文端：沿用 jieba 词级分词和 35,449-token 词表。
- 英文端：沿用 hw1 的 10,000-token byte-level BPE，并增加翻译特殊 token。
- 模型：Encoder-Decoder Transformer，包含因果自注意力和跨注意力。
- 解码：贪心搜索和重复惩罚。

```bash
cd translator-transformer
python -m pip install -r requirements.txt
curl -L -o data/zh-en.txt \
  https://raw.githubusercontent.com/Twilight-0419/Chinese-English-Translator/main/data/zh-en.txt
python -m src.prepare_data --overwrite
python -m src.train --device cuda
python -m src.main --text "你叫什么名字？"
```

## 模型权重

训练检查点体积较大，不提交到普通 Git 历史中。训练后可分别使用：

```text
poetry-transformer/checkpoints/poetry_transformer.pt
translator-transformer/checkpoints/translator_best.pt
```

如需分享权重，建议使用 GitHub Releases 或兼容 Git LFS 的存储。

## 目录结构

```text
.
├── poetry-transformer/
│   ├── data/
│   ├── models/
│   ├── src/
│   ├── tests/
│   └── README.md
└── translator-transformer/
    ├── data/
    ├── models/
    ├── src/
    ├── tests/
    └── README.md
```

各子项目保留了独立的依赖、测试、数据来源说明和第三方声明。
