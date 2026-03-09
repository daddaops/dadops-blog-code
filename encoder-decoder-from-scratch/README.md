# Encoder-Decoder from Scratch

Verified, runnable code from the [Encoder-Decoder from Scratch](https://dadops.dev/blog/encoder-decoder-from-scratch/) blog post.

## Scripts

- **lstm_seq2seq.py** — Original LSTM seq2seq with bottleneck context vector
- **attention_seq2seq.py** — Bahdanau attention added to solve the bottleneck
- **transformer_enc_dec.py** — Full transformer encoder-decoder with multi-head attention

## Run

```bash
pip install -r requirements.txt
python lstm_seq2seq.py
python attention_seq2seq.py
python transformer_enc_dec.py
```
