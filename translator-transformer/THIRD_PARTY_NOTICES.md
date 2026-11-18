# Third-Party Notices

## Chinese-English translation corpus

`data/zh-en.txt` is copied from the completed RNN translation project. It was
derived from seven OPUS corpora (ParaCrawl, OpenSubtitles, WikiMatrix, TED2013,
QED, wikimedia, and News-Commentary). Source-level details and known data quality
limitations are preserved in `data/RNN_DATA.md`.

OPUS does not provide one unified license for all upstream corpora. Review the
terms of OPUS and each original dataset before redistribution or commercial use.

## NLPDL-2026Fall hw1_bpe_and_lm

The following implementations are reused from the completed course assignment:

- Byte-level BPE trainer and tokenizer.
- RoPE, RMSNorm, SwiGLU, scaled dot-product attention, Linear, and Embedding.
- AdamW, cross entropy, learning-rate schedule, and gradient clipping.

The encoder-decoder Transformer, cross-attention, translation data packing,
training loop, checkpoint format, and command-line translator were added for
this project.
