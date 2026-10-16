"""
QC_tools/camera_stability.py

Measures frame-to-frame brightness stability, to verify that
camera.manual_settings actually pinned the exposure.

Why a separate check is needed: cv2's cap.set() returns True even for
properties the driver quietly ignores, and reading a property back can
return the value you just wrote regardless of whether it took effect.
Neither is proof. The only reliable evidence is empirical - point the
camera at an unchanging scene and see whether the pixels stop moving.

Usage:
    # Baseline first, with camera.manual_settings.enabled = false
    python -m QC_tools.camera_stability --seconds 30 --label auto

    # Then set enabled: true, tune the values, and re-run
    python -m QC_tools.camera_stability --seconds 30 --label locked

Keep the scene still for the whole run (product in place, nobody walking
past the light). Any brightness change measured is then the camera, not
the scene.

Reading the result: what matters is the spread (std dev and range) of
mean frame brightness, not its absolute level. A locked camera should
show a small, flat spread. A drifting one shows visible wander, and
often a slow ramp over the first seconds as auto-exposure hunts.
"""

import argparse
import statistics
import sys
import time

import cv2
import numpy as np

from backend.core.camera import Camera
from backend.core.config_loader import cfg
from backend.utils.logger import setup_logging, get_logger

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure camera brightness stability over time.")
    parser.add_argument("--seconds", type=float, default=30.0, help="How long to sample for.")
    parser.add_argument("--label", default="run", help="Name for this run, shown in the summary.")
    parser.add_argument("--warmup", type=float, default=3.0,
                        help="Seconds to discard at the start (lets the camera settle after opening).")
    args = parser.parse_args()

    setup_logging()

    manual = (cfg["camera"].get("manual_settings") or {})
    print()
    print(f"camera.manual_settings.enabled = {manual.get('enabled', False)}")
    if manual.get("enabled"):
        shown = {k: v for k, v in manual.items() if k != "enabled" and v is not None}
        print(f"  requested: {shown}")
    print("Check the log above for 'Camera manual settings applied' / 'NOT honoured'.")
    print()

    camera = Camera()
    time.sleep(max(0.0, args.warmup))

    means = []
    started = time.monotonic()
    last_report = started

    try:
        while time.monotonic() - started < args.seconds:
            frame = camera.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            means.append(float(np.mean(gray)))

            now = time.monotonic()
            if now - last_report >= 5.0:
                print(f"  {now - started:5.1f}s  frames={len(means):<5} "
                      f"current_mean_brightness={means[-1]:.2f}")
                last_report = now
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        try:
            camera.release()
        except Exception:
            pass

    if len(means) < 10:
        print("ERROR: not enough frames captured to judge stability.")
        return 1

    mean = statistics.mean(means)
    sd = statistics.pstdev(means)
    rng = max(means) - min(means)
    # Drift measured as first-fifth vs last-fifth average, which catches a
    # slow one-directional ramp that a raw std dev can understate.
    fifth = max(1, len(means) // 5)
    drift = statistics.mean(means[-fifth:]) - statistics.mean(means[:fifth])

    print()
    print("=" * 60)
    print(f"  {args.label}:  {len(means)} frames over {args.seconds:.0f}s")
    print(f"  mean brightness   {mean:7.2f}")
    print(f"  std dev           {sd:7.3f}")
    print(f"  min-max range     {rng:7.2f}   ({min(means):.2f} .. {max(means):.2f})")
    print(f"  start-to-end drift{drift:+7.2f}")
    print("=" * 60)
    print()
    print("  Compare an 'auto' run against a 'locked' run: the locked one should")
    print("  show a clearly smaller std dev, range and drift. If they look the")
    print("  same, the lock did not take - revisit camera.manual_settings")
    print("  .auto_exposure (see the tuning notes in config/default.yaml).")
    print()
    print("  Note this measures WHOLE-FRAME brightness, which is what exposure")
    print("  drift moves. It is not a focus or sharpness check.")

    return 0


if __name__ == "__main__":
    sys.exit(main())