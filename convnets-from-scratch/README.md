# Convolutional Neural Networks from Scratch

Code extracted from the DadOps blog post "Convolutional Neural Networks from Scratch".

## Scripts

| # | Script | Description |
|---|--------|-------------|
| 1 | `01_basic_conv2d.py` | Basic 2D convolution with Sobel edge detection demo |
| 2 | `02_multichannel_conv.py` | Multi-channel convolution with stride and padding |
| 3 | `03_max_pooling.py` | Max pooling layer |
| 4 | `04_lenet.py` | LeNet-style CNN architecture (forward pass) |
| 5 | `05_training.py` | Training utilities: softmax, cross-entropy, synthetic digit dataset |
| 6 | `06_batch_norm.py` | Batch normalization for convolutional layers |

## Requirements

```
numpy
```

Install with: `pip install -r requirements.txt`

## Running

Each script is self-contained and can be run independently:

```bash
python3 01_basic_conv2d.py
python3 02_multichannel_conv.py
python3 03_max_pooling.py
python3 04_lenet.py
python3 05_training.py
python3 06_batch_norm.py
```
