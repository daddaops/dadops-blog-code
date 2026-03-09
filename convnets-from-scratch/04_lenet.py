"""LeNet-style CNN architecture from scratch."""
import numpy as np

# --- Dependencies from earlier blocks ---

def conv2d_multi(x, kernels, biases, stride=1, padding=0):
    """Multi-channel convolution.
    x:       (C_in, H, W)
    kernels: (C_out, C_in, kH, kW)
    biases:  (C_out,)
    returns: (C_out, H_out, W_out)
    """
    c_out, c_in, k_h, k_w = kernels.shape
    _, h, w = x.shape

    if padding > 0:
        x = np.pad(x, ((0,0), (padding,padding), (padding,padding)))
        _, h, w = x.shape

    out_h = (h - k_h) // stride + 1
    out_w = (w - k_w) // stride + 1
    output = np.zeros((c_out, out_h, out_w))

    for f in range(c_out):
        for i in range(out_h):
            for j in range(out_w):
                si, sj = i * stride, j * stride
                patch = x[:, si:si+k_h, sj:sj+k_w]
                output[f, i, j] = np.sum(patch * kernels[f]) + biases[f]

    return output

def max_pool2d(x, pool_size=2):
    """Max pooling over spatial dimensions.
    x:       (C, H, W)
    returns: (C, H//pool_size, W//pool_size)
    """
    c, h, w = x.shape
    out_h = h // pool_size
    out_w = w // pool_size
    output = np.zeros((c, out_h, out_w))

    for ch in range(c):
        for i in range(out_h):
            for j in range(out_w):
                si, sj = i * pool_size, j * pool_size
                window = x[ch, si:si+pool_size, sj:sj+pool_size]
                output[ch, i, j] = np.max(window)

    return output

# --- Block 4: LeNet ---

def relu(x):
    return np.maximum(0, x)

class LeNet:
    def __init__(self):
        # Xavier initialization for stable training
        self.conv1_w = np.random.randn(6, 1, 5, 5) * np.sqrt(2.0 / 25)
        self.conv1_b = np.zeros(6)
        self.conv2_w = np.random.randn(16, 6, 5, 5) * np.sqrt(2.0 / 150)
        self.conv2_b = np.zeros(16)
        self.fc1_w = np.random.randn(256, 120) * np.sqrt(2.0 / 256)
        self.fc1_b = np.zeros(120)
        self.fc2_w = np.random.randn(120, 84) * np.sqrt(2.0 / 120)
        self.fc2_b = np.zeros(84)
        self.fc3_w = np.random.randn(84, 10) * np.sqrt(2.0 / 84)
        self.fc3_b = np.zeros(10)

    def forward(self, x):
        """Forward pass. x shape: (1, 28, 28)"""
        # Conv block 1
        x = conv2d_multi(x, self.conv1_w, self.conv1_b)  # (6, 24, 24)
        x = relu(x)
        x = max_pool2d(x)                                 # (6, 12, 12)

        # Conv block 2
        x = conv2d_multi(x, self.conv2_w, self.conv2_b)  # (16, 8, 8)
        x = relu(x)
        x = max_pool2d(x)                                 # (16, 4, 4)

        # Flatten and fully connected layers
        x = x.reshape(-1)                                  # (256,)
        x = relu(x @ self.fc1_w + self.fc1_b)             # (120,)
        x = relu(x @ self.fc2_w + self.fc2_b)             # (84,)
        x = x @ self.fc3_w + self.fc3_b                   # (10,) -- raw logits

        return x

if __name__ == "__main__":
    net = LeNet()
    dummy = np.random.randn(1, 28, 28)
    logits = net.forward(dummy)
    print("Logits shape:", logits.shape)  # (10,)
    print("Prediction:", np.argmax(logits))
