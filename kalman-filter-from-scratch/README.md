# Kalman Filter from Scratch

Verified, runnable code from the [Kalman Filter from Scratch](https://dadops.dev/blog/kalman-filter-from-scratch/) blog post.

## Scripts

- **data_setup.py** — Generate ground truth trajectory and noisy measurements
- **kalman_filter.py** — 1D Kalman filter for position/velocity tracking
- **kalman_gain.py** — Kalman gain convergence under different noise levels
- **kalman_2d.py** — 2D position+velocity tracking with velocity inference
- **ekf_radar.py** — Extended Kalman Filter for nonlinear radar tracking
- **sensor_fusion.py** — Fusing accelerometer (100 Hz) with GPS (1 Hz)

## Run

```bash
pip install -r requirements.txt
python data_setup.py
python kalman_filter.py
python kalman_gain.py
python kalman_2d.py
python ekf_radar.py
python sensor_fusion.py
```
