"""Statistical Anomaly Detection: Z-Score, MAD, and Grubbs' Test.

Three approaches to detecting anomalies assuming roughly Gaussian data:
- Z-score: flag points beyond k standard deviations
- MAD: robust variant using Median Absolute Deviation
- Grubbs' test: formal hypothesis test for single most extreme outlier

Applied to simulated server response times with injected spikes and gradual drift.
"""

def z_score_detector(data, threshold=3.0):
    """Flag points where |z-score| exceeds threshold."""
    mean = sum(data) / len(data)
    std = (sum((x - mean)**2 for x in data) / len(data)) ** 0.5
    return [i for i, x in enumerate(data) if abs((x - mean) / std) > threshold]

def mad_detector(data, threshold=3.5):
    """Modified z-score using Median Absolute Deviation (robust to outliers)."""
    sorted_d = sorted(data)
    median = sorted_d[len(sorted_d) // 2]
    mad = sorted(abs(x - median) for x in data)[len(data) // 2]
    # 0.6745 is the 75th percentile of the standard normal distribution
    modified_z = [0.6745 * (x - median) / mad if mad > 0 else 0 for x in data]
    return [i for i, mz in enumerate(modified_z) if abs(mz) > threshold]

def grubbs_test(data, alpha=0.05):
    """Test if the most extreme point is a significant outlier."""
    n = len(data)
    mean = sum(data) / n
    std = (sum((x - mean)**2 for x in data) / (n - 1)) ** 0.5
    # Find the point farthest from the mean
    max_idx = max(range(n), key=lambda i: abs(data[i] - mean))
    G = abs(data[max_idx] - mean) / std
    # Critical value approximation using t-distribution threshold
    # For large n, G_crit ~ sqrt((n-1)/n) * t_{alpha/(2n), n-2}
    t_crit = 2.5  # approximate for alpha=0.05, moderate n
    G_crit = ((n - 1) / n**0.5) * (t_crit**2 / (n - 2 + t_crit**2))**0.5
    return [max_idx] if G > G_crit else []

if __name__ == "__main__":
    # Simulate server response times: normal ~50ms, with injected anomalies
    import random
    random.seed(42)
    response_times = [random.gauss(50, 8) for _ in range(100)]
    response_times[30] = 150   # sudden spike
    response_times[70] = 180   # sudden spike
    for i in range(85, 95):    # gradual degradation
        response_times[i] = 50 + (i - 85) * 10 + random.gauss(0, 3)

    print("Z-score flags:", z_score_detector(response_times))
    print("MAD flags:    ", mad_detector(response_times))
    print("Grubbs flag:  ", grubbs_test(response_times))
