"""
Load Pattern Generators — ramp, sustained, burst, and soak patterns.

From: https://dadops.dev/blog/load-testing-ai-apis/
Code Block 4: "Load Patterns That Reveal Real Problems"

Defines four load pattern generators that yield (timestamp, target_concurrency)
pairs, plus a build_test_plan() composer.
"""

import time


def ramp_pattern(peak_concurrency, duration_sec, steps=20):
    """Linearly ramp from 1 to peak over duration."""
    step_time = duration_sec / steps
    for i in range(steps + 1):
        target = max(1, int(peak_concurrency * i / steps))
        yield time.time() + i * step_time, target


def sustained_pattern(concurrency, duration_sec, interval=1.0):
    """Hold constant concurrency for duration."""
    steps = int(duration_sec / interval)
    for i in range(steps):
        yield time.time() + i * interval, concurrency


def burst_pattern(base, multiplier, duration_sec, burst_every=10):
    """Base load with periodic bursts of multiplier * base."""
    t = 0
    while t < duration_sec:
        in_burst = (t % burst_every) >= (burst_every * 0.7)
        target = base * multiplier if in_burst else base
        yield time.time() + t, target
        t += 0.5


def soak_pattern(concurrency, duration_sec, interval=1.0):
    """Constant load — pair with latency monitoring for drift."""
    steps = int(duration_sec / interval)
    for i in range(steps):
        yield time.time() + i * interval, concurrency


def build_test_plan(phases):
    """Compose patterns into a full test plan.

    phases = [
        ("ramp", {"peak_concurrency": 64, "duration_sec": 30}),
        ("sustained", {"concurrency": 48, "duration_sec": 120}),
        ("burst", {"base": 32, "multiplier": 5, "duration_sec": 60}),
    ]
    """
    plan = []
    for name, kwargs in phases:
        gen = {"ramp": ramp_pattern, "sustained": sustained_pattern,
               "burst": burst_pattern, "soak": soak_pattern}[name]
        plan.extend(gen(**kwargs))
    return plan


if __name__ == "__main__":
    print("=== Load Pattern Generators Demo ===\n")

    # Test each pattern
    print("1. Ramp Pattern (peak=64, 10s, 10 steps):")
    base_time = None
    for ts, target in ramp_pattern(64, 10, steps=10):
        if base_time is None:
            base_time = ts
        print(f"   t={ts - base_time:5.1f}s  concurrency={target}")

    print("\n2. Sustained Pattern (concurrency=48, 5s):")
    base_time = None
    for ts, target in sustained_pattern(48, 5, interval=1.0):
        if base_time is None:
            base_time = ts
        print(f"   t={ts - base_time:5.1f}s  concurrency={target}")

    print("\n3. Burst Pattern (base=32, 5x, 20s, burst every 10s):")
    base_time = None
    count = 0
    for ts, target in burst_pattern(32, 5, 20, burst_every=10):
        if base_time is None:
            base_time = ts
        count += 1
        in_burst = "BURST" if target > 32 else "base"
        print(f"   t={ts - base_time:5.1f}s  concurrency={target:>4} [{in_burst}]")
    print(f"   Total steps: {count}")

    print("\n4. Soak Pattern (concurrency=24, 5s):")
    base_time = None
    for ts, target in soak_pattern(24, 5, interval=1.0):
        if base_time is None:
            base_time = ts
        print(f"   t={ts - base_time:5.1f}s  concurrency={target}")

    # Test build_test_plan composition
    print("\n5. Composed Test Plan:")
    phases = [
        ("ramp", {"peak_concurrency": 64, "duration_sec": 5, "steps": 5}),
        ("sustained", {"concurrency": 48, "duration_sec": 3}),
        ("burst", {"base": 32, "multiplier": 5, "duration_sec": 5}),
    ]
    plan = build_test_plan(phases)
    print(f"   Total steps in plan: {len(plan)}")
    print(f"   Concurrency range: {min(t for _, t in plan)} - {max(t for _, t in plan)}")

    # Verify burst pattern math: 30% of each period is burst
    print("\n=== Burst Pattern Verification ===")
    print("   Blog says burst at 70% of each period (last 30% is burst)")
    burst_steps = sum(1 for _, t in burst_pattern(10, 5, 20, burst_every=10) if t > 10)
    total_steps = sum(1 for _ in burst_pattern(10, 5, 20, burst_every=10))
    print(f"   Burst steps: {burst_steps}/{total_steps} "
          f"({burst_steps/total_steps*100:.0f}%)")
