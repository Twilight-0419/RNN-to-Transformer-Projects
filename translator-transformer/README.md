# Chinese-English Translator Transformer

这是一个中文到英文的 Transformer 神经机器翻译项目，结合了两个已有任务：

- 中文数据清洗、jieba 分词、平行语料和中文词表来自 RNN 翻译器。
- 英文目标端使用 `hw1_bpe_and_lm` 的 byte-level BPE 分词器。
- 模型复用作业中的 RoPE、RMSNorm、SwiGLU、缩放点积注意力和 AdamW。

项目新增了编码器-解码器 Transformer、跨注意力、翻译数据打包、训练、
检查点和命令行推理。

## 模型结构

```text
中文句子
-> jieba 词级分词
-> Encoder Self-Attention
-> Encoder Memory
-> Decoder Causal Self-Attention
-> Decoder Cross-Attention
-> 英文 BPE token 概率
-> 贪心解码与重复惩罚
```

中文词表沿用 RNN 项目的 35,449 个 token。英文使用作业的 10,000-token
byte-level BPE，并增加 `<pad>`、`<sos>`、`<eos>`，目标词表共 10,003。
BPE 基于字节，因此英文侧不会出现 RNN 词表里的 `<unk>`。

## 目录结构

```text
Chinese-English-Translator-Transformer/
├── data/
│   ├── zh-en.txt (从原 RNN 仓库下载)
│   └── RNN_DATA.md
├── models/
│   ├── rnn_vocab.pt
│   ├── bpe_vocab.json
│   └── bpe_merges.txt
├── src/
│   ├── bpe.py
│   ├── bpe_tokenizer.py
│   ├── data.py
│   ├── main.py
│   ├── nmt_model.py
│   ├── optimizer.py
│   ├── prepare_data.py
│   ├── text.py
│   ├── train.py
│   ├── transformer_layers.py
│   └── translator.py
└── tests/
```

## 安装

```bash
python -m pip install -r requirements.txt
```

## 准备数据

138 万句对语料较大，新仓库不重复保存。先从已发布的 RNN 翻译仓库下载：

```bash
curl -L -o data/zh-en.txt \
  https://raw.githubusercontent.com/Twilight-0419/Chinese-English-Translator/main/data/zh-en.txt
```

处理完整 138 万句对：

```bash
python -m src.prepare_data --overwrite
```

输出位于：

```text
data/processed/train_source.npy
data/processed/train_target.npy
data/processed/val_source.npy
data/processed/val_target.npy
data/processed/test_source.npy
data/processed/test_target.npy
data/processed/metadata.json
```

快速流程验证可以只处理 3,000 句对：

```bash
python -m src.prepare_data --max-pairs 3000 --output-dir data/smoke --overwrite
```

## 训练

RTX 3090 24GB 推荐从以下配置开始：

```bash
python -m src.train \
  --data-dir data/processed \
  --device cuda \
  --d-model 256 \
  --num-encoder-layers 4 \
  --num-decoder-layers 4 \
  --num-heads 8 \
  --d-ff 1024 \
  --batch-size 128 \
  --epochs 5 \
  --lr 3e-4 \
  --min-lr 3e-5 \
  --warmup-steps 1000 \
  --weight-decay 0.01 \
  --grad-clip 1.0 \
  --log-interval 50 \
  --val-interval 1000 \
  --max-val-batches 100 \
  --early-stop-patience 3 \
  --num-workers 4 \
  --checkpoint-dir checkpoints
```

显存不足时先把 `--batch-size` 改成 64 或 32。

训练自动保存：

```text
checkpoints/translator_best.pt
checkpoints/translator_latest.pt
checkpoints/translator_transformer.pt
```

优先使用 `translator_best.pt` 做推理。

### 小数据冒烟训练

先执行前面的 3,000 句对数据准备，再运行：

```bash
python -m src.train \
  --data-dir data/smoke \
  --device cpu \
  --d-model 64 \
  --num-encoder-layers 1 \
  --num-decoder-layers 1 \
  --num-heads 4 \
  --d-ff 128 \
  --batch-size 8 \
  --max-steps 20 \
  --warmup-steps 1 \
  --log-interval 1 \
  --val-interval 10 \
  --max-val-batches 2 \
  --num-workers 0 \
  --checkpoint-dir checkpoints/smoke
```

## 翻译

交互式：

```bash
python -m src.main --checkpoint checkpoints/translator_best.pt
```

单句：

```bash
python -m src.main \
  --checkpoint checkpoints/translator_best.pt \
  --device cuda \
  --text "我爱你。"
```

Python 调用：

```python
from src.translator import TransformerTranslator

translator = TransformerTranslator("checkpoints/translator_best.pt")
print(translator.translate("你叫什么名字？"))
```

## 测试

```bash
python -m unittest discover -s tests -v
```

## 说明

- 语料由多个 OPUS 数据源组成，仍有对齐噪声。
- 数据经过筛选，主要覆盖 6–19 个汉字的短句。
- 当前解码器使用贪心搜索加重复惩罚，不包含 beam search。
- 英文 BPE 来自 TinyStories 数据集，目标端没有 OOV，但领域分布与 OPUS
  翻译语料不完全一致。
