"""Training loop with synthetic digit data and numerical gradient descent."""
import numpy as np

# --- Dependencies from earlier blocks ---

def conv2d_multi(x, kernels, biases, stride=1, padding=0):
    """Multi-channel convolution."""
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
    """Max pooling over spatial dimensions."""
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

def relu(x):
    return np.maximum(0, x)

class LeNet:
    def __init__(self):
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
        x = conv2d_multi(x, self.conv1_w, self.conv1_b)
        x = relu(x)
        x = max_pool2d(x)
        x = conv2d_multi(x, self.conv2_w, self.conv2_b)
        x = relu(x)
        x = max_pool2d(x)
        x = x.reshape(-1)
        x = relu(x @ self.fc1_w + self.fc1_b)
        x = relu(x @ self.fc2_w + self.fc2_b)
        x = x @ self.fc3_w + self.fc3_b
        return x

# --- Block 5: Training utilities ---

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def cross_entropy_loss(logits, target):
    probs = softmax(logits)
    return -np.log(probs[target] + 1e-10)

def make_synthetic_digit(label, size=28):
    """Generate a simple synthetic image for a digit class."""
    img = np.zeros((size, size), dtype=np.float32)
    rng = np.random
    cx, cy = size // 2, size // 2

    if label == 0:     # circle
        for y in range(size):
            for x in range(size):
                if 8 <= ((x-cx)**2 + (y-cy)**2)**0.5 <= 11:
                    img[y, x] = 1.0
    elif label == 1:   # vertical line
        img[4:-4, cx-1:cx+1] = 1.0
    elif label == 2:   # horizontal line
        img[cy-1:cy+1, 4:-4] = 1.0
    elif label == 3:   # diagonal (top-left to bottom-right)
        for k in range(-1, 2):
            np.fill_diagonal(img[max(0,k):, max(0,-k):], 1.0)
    elif label == 4:   # cross (+)
        img[cy-1:cy+1, 4:-4] = 1.0
        img[4:-4, cx-1:cx+1] = 1.0
    elif label == 5:   # X shape
        for k in range(-1, 2):
            np.fill_diagonal(img[max(0,k):, max(0,-k):], 1.0)
            np.fill_diagonal(np.fliplr(img)[max(0,k):, max(0,-k):], 1.0)
    elif label == 6:   # square
        img[5:23, 5:7] = img[5:23, 21:23] = 1.0
        img[5:7, 5:23] = img[21:23, 5:23] = 1.0
    elif label == 7:   # triangle (top)
        for row in range(6, 24):
            half_w = (row - 6) * 10 // 18
            img[row, cx-half_w:cx+half_w+1] = 1.0
    elif label == 8:   # diamond
        for row in range(size):
            dist = abs(row - cy)
            half_w = max(0, 10 - dist)
            if half_w > 0:
                img[row, cx-half_w:cx+half_w+1] = 1.0
    elif label == 9:   # small dot
        img[cy-3:cy+3, cx-3:cx+3] = 1.0

    # Add slight noise for variety
    img += rng.randn(size, size) * 0.05
    return np.clip(img, 0, 1)

if __name__ == "__main__":
    # Generate dataset: 100 samples per class
    X_train, y_train = [], []
    for label in range(10):
        for _ in range(100):
            X_train.append(make_synthetic_digit(label))
            y_train.append(label)

    X_train = np.array(X_train).reshape(-1, 1, 28, 28)
    y_train = np.array(y_train)

    # Shuffle
    perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]
    print(f"Dataset: {len(X_train)} images, {len(set(y_train))} classes")

    # Quick forward pass test with loss
    net = LeNet()
    logits = net.forward(X_train[0])
    loss = cross_entropy_loss(logits, y_train[0])
    print(f"Initial loss on first sample: {loss:.4f}")
    print(f"Prediction: {np.argmax(logits)}, True label: {y_train[0]}")
