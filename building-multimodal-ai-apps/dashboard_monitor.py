"""
Dashboard monitoring with VLM: periodic screenshots analyzed by Claude.

Blog post: https://dadops.dev/blog/building-multimodal-ai-apps/
Code Block 4.

Requires: anthropic, chromium browser.
NOTE: Requires ANTHROPIC_API_KEY environment variable and Chromium installed.
"""
import base64
import time
import subprocess
from datetime import datetime
from pathlib import Path


# ── Code Block 4: Dashboard Monitoring ──

def capture_dashboard(url: str, output_path: str) -> str:
    """Capture a dashboard screenshot using a headless browser."""
    subprocess.run([
        "chromium", "--headless", "--disable-gpu",
        f"--screenshot={output_path}", f"--window-size=1920,1080",
        url
    ], capture_output=True, timeout=30)
    return output_path


def monitor_dashboard(url: str, interval_minutes: int = 15):
    """Periodically screenshot a dashboard and check for anomalies."""
    import anthropic

    previous_summary = None

    while True:
        screenshot_path = f"/tmp/dashboard_{int(time.time())}.png"
        capture_dashboard(url, screenshot_path)

        image_bytes = Path(screenshot_path).read_bytes()
        b64_image = base64.b64encode(image_bytes).decode("utf-8")

        context = f"Previous check summary: {previous_summary}" if previous_summary else ""

        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-5-20250514",
            max_tokens=800,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png",
                        "data": b64_image
                    }},
                    {"type": "text", "text": (
                        f"You are monitoring a dashboard. {context}\n\n"
                        f"1. Briefly summarize the current state of all visible metrics.\n"
                        f"2. Flag any anomalies, spikes, or concerning trends.\n"
                        f"3. Rate overall system health: HEALTHY / WARNING / CRITICAL.\n"
                        f"4. If WARNING or CRITICAL, explain what needs attention."
                    )}
                ]
            }]
        )
        result = response.content[0].text
        previous_summary = result[:500]

        if "CRITICAL" in result or "WARNING" in result:
            print(f"[{datetime.now():%H:%M}] ALERT:\n{result}")
            # send_slack_alert(result)  # Wire up your alerting

        Path(screenshot_path).unlink()  # Clean up
        time.sleep(interval_minutes * 60)


if __name__ == "__main__":
    print("=== Dashboard Monitor ===\n")
    print("This script requires ANTHROPIC_API_KEY and Chromium to run.")
    print("Usage: monitor_dashboard('https://your-dashboard.com', interval_minutes=15)")
    print("No self-tests available without API key and browser.")
