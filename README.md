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
