# Tang Poetry Generator Transformer

这是一个把已有 RNN 唐诗项目的数据与分词方式，和 `hw1_bpe_and_lm`
中从零实现的 Transformer 结合起来的唐诗生成器。

仓库不附带正式训练权重。请先完成一次训练，`checkpoints/poetry_transformer.pt`
生成后即可调用命令行生成器。

## 复用了什么

- `data/tang_poetry_corpus.txt`：沿用 RNN 项目的清洗后《全唐诗》语料。
- `models/preprocessing.pt`：可选的 RNN jieba 词级分词结果。文件缺失时，
  数据准备脚本会直接从语料重新构建高频词表。
- `src/transformer_model.py`：沿用作业中的 RoPE、RMSNorm、SwiGLU、
  因果多头自注意力和 `TransformerLM`，模型实现未改写。
- `src/optimizer.py`：沿用作业中的 AdamW、交叉熵、梯度裁剪和学习率调度。

RNN 的原始词表有 338,825 个词。直接照搬会让 Transformer 的嵌入层和输出层
过大，因此数据准备脚本默认按 RNN `corpus_idx` 中的词频，保留前 12,000 个
高频 token，并把低频 token 映射为 `<unk>`。这只是压缩模型词表，语料和
jieba 分词方式仍来自 RNN 项目。

## 项目结构

```text
Tang-Poetry-Generator-Transformer/
├── data/
│   └── tang_poetry_corpus.txt
├── models/
│   └── (可选) preprocessing.pt
├── src/
│   ├── data.py
│   ├── main.py
│   ├── optimizer.py
│   ├── poetry_generator.py
│   ├── prepare_data.py
│   ├── tokenizer.py
│   ├── train.py
│   └── transformer_model.py
├── tests/
├── requirements.txt
└── README.md
```

## 安装

建议在项目根目录创建独立环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 第一次训练

### 1. 生成紧凑词表和训练数据

如果想复用 RNN 项目的完整预处理文件，先下载：

```powershell
curl.exe -L -o models\preprocessing.pt `
  https://raw.githubusercontent.com/Twilight-0419/Tang-Poetry-Generator/main/models/preprocessing.pt
```

不下载也可以直接运行下一步，程序会从 `data/tang_poetry_corpus.txt`
重新分词并构建 12,000-token 高频词表。

```powershell
python -m src.prepare_data
```

该命令会生成：

```text
data/train.bin
data/val.bin
data/vocab.json
data/dataset_info.json
```

如果想改变词表大小，例如保留 16,000 个高频词：

```powershell
python -m src.prepare_data --vocab-size 16000 --overwrite
```

如果以后修改了语料，不想继续使用旧的 `preprocessing.pt`：

```powershell
python -m src.prepare_data --force-retokenize --overwrite
```

### 2. 从头训练约两小时

面向 RTX 3090 24GB 的推荐配置如下。数据只有约 215 万 token，不建议直接
照搬 TinyStories 的 40,000 步配置。`--max-iters` 是上限，实际训练应在约
两小时后手动停止，或者根据验证集 loss 提前停止。

```bash
python -m src.train \
  --device cuda \
  --context-length 256 \
  --d-model 768 \
  --num-layers 8 \
  --num-heads 12 \
  --d-ff 2048 \
  --batch-size 32 \
  --max-iters 30000 \
  --lr 2e-4 \
  --min-lr 2e-5 \
  --warmup-iters 500 \
  --weight-decay 0.05 \
  --grad-clip 1.0 \
  --log-interval 20 \
  --val-interval 250 \
  --save-interval 250 \
  --early-stop-patience 3 \
  --checkpoint-dir checkpoints/poetry_2h
```

如果出现显存不足，把 `--batch-size 32` 改为 `--batch-size 16`。

Windows PowerShell 中的相同命令：

```powershell
python -m src.train `
  --device cuda `
  --context-length 256 `
  --d-model 768 `
  --num-layers 8 `
  --num-heads 12 `
  --d-ff 2048 `
  --batch-size 32 `
  --max-iters 30000 `
  --lr 2e-4 `
  --min-lr 2e-5 `
  --warmup-iters 500 `
  --weight-decay 0.05 `
  --early-stop-patience 3 `
  --checkpoint-dir checkpoints/poetry_2h
```

CPU 可以先跑一个较小配置：

```powershell
python -m src.train `
  --device cpu `
  --context-length 128 `
  --d-model 256 `
  --num-layers 4 `
  --num-heads 8 `
  --d-ff 768 `
  --batch-size 32 `
  --max-iters 5000 `
  --lr 3e-4 `
  --min-lr 3e-5 `
  --warmup-iters 200 `
  --checkpoint-dir checkpoints
```

训练中每 `--save-interval` 步会保存：

```text
checkpoints/poetry_2h/checkpoint_latest.pt
```

验证集出现新低时还会保存：

```text
checkpoints/poetry_2h/checkpoint_best.pt
```

训练结束后会保存最终模型：

```text
checkpoints/poetry_2h/poetry_transformer.pt
```

训练两小时后可以按 `Ctrl+C` 停止。最近一次每 250 步保存的检查点仍然有效。
`--early-stop-patience 3` 会在连续三次验证都没有改善时自动停止。通常应优先
使用 `checkpoint_best.pt`，因为 `checkpoint_latest.pt` 可能是过拟合后的版本。

### 3. 继续训练

如果训练中断，可以从最近检查点继续。模型结构参数必须与检查点完全一致；
`--max-iters` 要设置为新的总步数，而不是“再从当前检查点训练多少步”：

```bash
python -m src.train \
  --device cuda \
  --context-length 256 \
  --d-model 768 \
  --num-layers 8 \
  --num-heads 12 \
  --d-ff 2048 \
  --batch-size 32 \
  --max-iters 36000 \
  --resume checkpoints/poetry_2h/checkpoint_best.pt
```

恢复训练时，模型宽度、层数、上下文长度等结构参数必须与检查点保持一致。

## 生成诗歌

交互式运行：

```powershell
python -m src.main
```

程序会提示输入起始字或句子以及生成词数。也可以一次传入参数：

```powershell
python -m src.main --start 春 --length 30
```

控制生成风格：

```powershell
python -m src.main `
  --start 明月 `
  --length 40 `
  --temperature 0.8 `
  --top-p 0.9 `
  --repetition-penalty 2.0 `
  --line-width 12 `
  --max-extra-tokens 32
```

- `temperature` 越低越稳定，越高越随机。
- `top-p` 控制候选词累计概率。
- `top-k` 大于 0 时进一步限制候选词数量。
- `repetition-penalty` 会降低最近出现 token 的分数。
- `line-width` 是展示时的自动换行宽度。
- `length` 是最少新增 token 数。达到该长度后，程序会继续生成，直到输出
  中文句号 `。`。
- `max-extra-tokens` 是等待句号时最多允许的额外 token 数，默认 32。

代码调用：

```python
from src.poetry_generator import TransformerPoetryGenerator

generator = TransformerPoetryGenerator("checkpoints/poetry_2h/checkpoint_best.pt")
poem = generator.generate(
    start_text="春",
    poem_length=30,
    temperature=0.8,
    top_p=0.9,
    max_extra_tokens=32,
)
print(poem)
```

兼容 RNN 项目旧入口的调用也保留了：

```python
from src.poetry_generator import Generating_Poetry

print(Generating_Poetry("春", 30))
```

## 测试

```powershell
python -m unittest discover -s tests -v
```

## 说明

这里的 `length` 是最少新增词 token 数，不是最终汉字的精确数量，因为
jieba 可能把“春风”“明月”等切成一个 token。达到长度后，生成器会等待
`。` 再停止，因此最终结果可能比 `length` 稍长。若达到额外 token 上限时
仍未生成句号，程序会在末尾补一个 `。` 后停止。

模型权重文件较大，默认不提交到 Git。数据和第三方来源说明见
`THIRD_PARTY_NOTICES.md`。
