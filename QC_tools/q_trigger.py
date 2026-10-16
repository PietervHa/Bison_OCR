"""
Automated trigger for long-running vision tests.

Two trigger methods are supported:

1. "keyboard" (legacy): simulates a global OS-level 'q' keypress at a
   configurable interval, picked up by backend/main.py's manual trigger loop
   (`keyboard.wait("q")`). Because this is a *global* OS keystroke, it lands
   in whatever window currently has focus -- tab away to code or browse while
   a long test runs and "q" gets typed there too.

2. "tcp" (recommended for local testing): connects directly to the app's TCP
   trigger server (the same one used for PLC integration, see
   backend/core/tcp_trigger_server.py) and sends the trigger byte over a
   local socket. Nothing is sent to the OS/keyboard, so it never interferes
   with whatever window you're using.

   To use "tcp" mode, enable the trigger server in config/default.yaml:

       trigger:
         enabled: true
         host: "127.0.0.1"   # keep it local-only unless you need otherwise

   ... and run the main app as usual. It will start listening on
   trigger.port (5001 by default) in addition to the keyboard listener.

Stopping the run:
   By default the run is bounded by --hours/max_hours (capped at 8h). You can
   instead (or additionally) bound it by a fixed number of triggers using
   --max-triggers/max_triggers, so you don't have to work out how many hours
   correspond to the number of triggers you actually want. Whichever limit is
   hit first stops the run. If --max-triggers is given without --hours, the
   time limit is effectively disabled (set to the 8h ceiling) so the trigger
   count is what actually governs the run.

Standalone usage:
   python -m QC_tools.q_trigger              # tcp mode (default)
   python -m QC_tools.q_trigger --delay 1 --interval 1 --hours 0.33 ->example of modified command
   python -m QC_tools.q_trigger --delay 1 --interval 1 --max-triggers 100 ->stop after exactly 100 triggers
   python -m QC_tools.q_trigger --mode keyboard

From the main application:
   start_q_trigger(mode="tcp", ...)
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading
import time
from typing import Callable, Optional

import keyboard

from backend.core.config_loader import cfg

log = logging.getLogger(__name__)

DEFAULT_DELAY_S = 30.0
DEFAULT_INTERVAL_S = 1.0
DEFAULT_MAX_HOURS = 8.0
DEFAULT_MAX_TRIGGERS: Optional[int] = None  # no limit by default -- bounded by max_hours instead


def _parse_trigger_byte(value) -> int:
    """Parse a trigger byte from config, accepting '0x01', '1', or an int."""
    if isinstance(value, str):
        try:
            return int(value, 0)
        except ValueError:
            log.warning("Invalid trigger_byte %r in config, falling back to 0x01", value)
            return 0x01
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0x01


_trig_cfg = cfg.get("trigger", {}) if isinstance(cfg, dict) else {}
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = int(_trig_cfg.get("port", 5001))
DEFAULT_TRIGGER_BYTE = _parse_trigger_byte(_trig_cfg.get("trigger_byte", "0x01"))
DEFAULT_TCP_CONNECT_TIMEOUT_S = 5.0
DEFAULT_TCP_RESPONSE_TIMEOUT_S = float(_trig_cfg.get("timeout_s", 5.0))


def run_q_trigger(
    stop_event: threading.Event,
    *,
    delay_s: float = DEFAULT_DELAY_S,
    interval_s: float = DEFAULT_INTERVAL_S,
    max_hours: float = DEFAULT_MAX_HOURS,
    max_triggers: Optional[int] = DEFAULT_MAX_TRIGGERS,
    can_trigger: Optional[Callable[[], bool]] = None,
) -> None:
    """Send global Q keypresses until stopped or a maximum limit is reached.

    The run stops on whichever limit is hit first: max_hours (capped at 8h)
    or max_triggers (total number of Q presses actually sent), if set.

    Legacy mode: this simulates an OS-level keystroke and will land in
    whatever window has focus. Prefer run_q_trigger_tcp for local testing.
    """
    delay_s = max(0.0, float(delay_s))
    interval_s = max(0.05, float(interval_s))
    max_hours = min(max(0.0, float(max_hours)), 8.0)
    max_duration_s = max_hours * 3600.0
    max_triggers = int(max_triggers) if max_triggers is not None and max_triggers > 0 else None

    log.info(
        "Automated Q trigger (keyboard) enabled: first trigger in %.1fs, interval %.2fs, "
        "max duration %.2fh, max triggers %s",
        delay_s,
        interval_s,
        max_hours,
        max_triggers if max_triggers is not None else "unlimited",
    )

    if stop_event.wait(delay_s):
        log.info("Automated Q trigger stopped before first trigger")
        return

    started_at = time.monotonic()
    trigger_count = 0

    while not stop_event.is_set():
        elapsed = time.monotonic() - started_at
        if elapsed >= max_duration_s:
            break
        if max_triggers is not None and trigger_count >= max_triggers:
            log.info("Automated Q trigger reached max_triggers=%d, stopping", max_triggers)
            break

        if can_trigger is None or can_trigger():
            try:
                keyboard.press("q")
                trigger_count += 1
                log.info("Automated Q trigger #%d", trigger_count)
            except Exception:
                log.exception("Failed to send automated Q trigger")
        else:
            log.debug("Skipping automated Q trigger because vision is busy")

        if max_triggers is not None and trigger_count >= max_triggers:
            log.info("Automated Q trigger reached max_triggers=%d, stopping", max_triggers)
            break

        # Keep the interval measured between trigger attempts rather than
        # adding it on top of the keypress/logging overhead.
        if stop_event.wait(interval_s):
            return

    log.info(
        "Automated Q trigger finished after %.2fh (%d Q presses)",
        min((time.monotonic() - started_at) / 3600.0, max_hours),
        trigger_count,
    )


def _connect_tcp(host: str, port: int, connect_timeout_s: float) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=connect_timeout_s)
    sock.settimeout(None)
    return sock


def run_q_trigger_tcp(
    stop_event: threading.Event,
    *,
    host: str = DEFAULT_TCP_HOST,
    port: int = DEFAULT_TCP_PORT,
    trigger_byte: int = DEFAULT_TRIGGER_BYTE,
    delay_s: float = DEFAULT_DELAY_S,
    interval_s: float = DEFAULT_INTERVAL_S,
    max_hours: float = DEFAULT_MAX_HOURS,
    max_triggers: Optional[int] = DEFAULT_MAX_TRIGGERS,
    can_trigger: Optional[Callable[[], bool]] = None,
    connect_timeout_s: float = DEFAULT_TCP_CONNECT_TIMEOUT_S,
    response_timeout_s: float = DEFAULT_TCP_RESPONSE_TIMEOUT_S,
) -> None:
    """Send trigger bytes over TCP until stopped or a maximum limit is reached.

    The run stops on whichever limit is hit first: max_hours (capped at 8h)
    or max_triggers (total number of triggers actually sent), if set.

    Talks to the same TCPTriggerServer used for PLC integration
    (backend/core/tcp_trigger_server.py). Requires `trigger.enabled: true` in
    config/default.yaml and the main application to be running. Unlike
    keyboard mode, nothing is sent to the OS -- this only ever touches a
    local socket, so it never steals focus or types into other windows.
    """
    delay_s = max(0.0, float(delay_s))
    interval_s = max(0.05, float(interval_s))
    max_hours = min(max(0.0, float(max_hours)), 8.0)
    max_duration_s = max_hours * 3600.0
    max_triggers = int(max_triggers) if max_triggers is not None and max_triggers > 0 else None

    log.info(
        "Automated TCP trigger enabled: target %s:%d, first trigger in %.1fs, interval %.2fs, "
        "max duration %.2fh, max triggers %s",
        host,
        port,
        delay_s,
        interval_s,
        max_hours,
        max_triggers if max_triggers is not None else "unlimited",
    )

    if stop_event.wait(delay_s):
        log.info("Automated TCP trigger stopped before first trigger")
        return

    started_at = time.monotonic()
    trigger_count = 0
    sock: Optional[socket.socket] = None

    try:
        while not stop_event.is_set():
            elapsed = time.monotonic() - started_at
            if elapsed >= max_duration_s:
                break
            if max_triggers is not None and trigger_count >= max_triggers:
                log.info("Automated TCP trigger reached max_triggers=%d, stopping", max_triggers)
                break

            if can_trigger is None or can_trigger():
                try:
                    if sock is None:
                        sock = _connect_tcp(host, port, connect_timeout_s)
                        log.info("Connected to TCP trigger server at %s:%d", host, port)

                    sock.sendall(bytes([trigger_byte]))
                    sock.settimeout(response_timeout_s)
                    try:
                        response = sock.recv(64)
                        trigger_count += 1
                        log.info(
                            "Automated TCP trigger #%d -> %s",
                            trigger_count,
                            response.decode(errors="replace").strip() or "<empty>",
                        )
                    except socket.timeout:
                        trigger_count += 1
                        log.warning(
                            "Automated TCP trigger #%d sent, no response within %.1fs",
                            trigger_count,
                            response_timeout_s,
                        )
                    finally:
                        sock.settimeout(None)
                except (OSError, ConnectionError) as exc:
                    log.warning("TCP trigger connection issue (%s); will reconnect next attempt", exc)
                    try:
                        if sock is not None:
                            sock.close()
                    except Exception:
                        pass
                    sock = None
            else:
                log.debug("Skipping automated TCP trigger because vision is busy")

            if max_triggers is not None and trigger_count >= max_triggers:
                log.info("Automated TCP trigger reached max_triggers=%d, stopping", max_triggers)
                break

            if stop_event.wait(interval_s):
                return
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass

    log.info(
        "Automated TCP trigger finished after %.2fh (%d triggers sent)",
        min((time.monotonic() - started_at) / 3600.0, max_hours),
        trigger_count,
    )


def start_q_trigger(
    *,
    mode: str = "keyboard",
    delay_s: float = DEFAULT_DELAY_S,
    interval_s: float = DEFAULT_INTERVAL_S,
    max_hours: float = DEFAULT_MAX_HOURS,
    max_triggers: Optional[int] = DEFAULT_MAX_TRIGGERS,
    can_trigger: Optional[Callable[[], bool]] = None,
    host: str = DEFAULT_TCP_HOST,
    port: int = DEFAULT_TCP_PORT,
    trigger_byte: int = DEFAULT_TRIGGER_BYTE,
) -> tuple[threading.Thread, threading.Event]:
    """Start the automated trigger in a daemon thread.

    mode="keyboard" (default, backward compatible): simulates a global 'q'
        keypress -- lands in whatever window has focus.
    mode="tcp": sends trigger bytes to the app's TCP trigger server instead;
        never touches the OS keyboard.

    max_triggers: optional cap on the total number of triggers to send. Set
        this instead of fiddling with max_hours when you just want "send N
        triggers and stop" -- whichever limit (time or count) is hit first
        wins.
    """
    stop_event = threading.Event()

    if mode == "tcp":
        target = run_q_trigger_tcp
        kwargs = {
            "stop_event": stop_event,
            "host": host,
            "port": port,
            "trigger_byte": trigger_byte,
            "delay_s": delay_s,
            "interval_s": interval_s,
            "max_hours": max_hours,
            "max_triggers": max_triggers,
            "can_trigger": can_trigger,
        }
    elif mode == "keyboard":
        target = run_q_trigger
        kwargs = {
            "stop_event": stop_event,
            "delay_s": delay_s,
            "interval_s": interval_s,
            "max_hours": max_hours,
            "max_triggers": max_triggers,
            "can_trigger": can_trigger,
        }
    else:
        raise ValueError(f"Unknown q_trigger mode: {mode!r} (expected 'keyboard' or 'tcp')")

    thread = threading.Thread(
        target=target,
        kwargs=kwargs,
        daemon=True,
        name="automated_q_trigger",
    )
    thread.start()
    return thread, stop_event


def main() -> None:
    parser = argparse.ArgumentParser(description="Send automated triggers for a long-running vision test.")
    parser.add_argument(
        "--mode",
        choices=["keyboard", "tcp"],
        default="tcp",
        help=(
            "Trigger method. 'tcp' (default) talks to the app's TCP trigger server over a local "
            "socket and never touches your keyboard/window focus. 'keyboard' simulates a global "
            "Q keypress (legacy behavior)."
        ),
    )
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_S, help="Seconds before the first trigger")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S, help="Seconds between triggers")
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help=(
            "Maximum runtime, capped at 8 hours. If omitted and --max-triggers is set, the time "
            "limit is effectively disabled (8h ceiling) so --max-triggers governs the run instead."
        ),
    )
    parser.add_argument(
        "--max-triggers",
        type=int,
        default=DEFAULT_MAX_TRIGGERS,
        help=(
            "Stop after sending exactly this many triggers, regardless of elapsed time. Combine "
            "with --interval to control the pace. Omit for no cap (time-based limit only)."
        ),
    )
    parser.add_argument("--host", type=str, default=DEFAULT_TCP_HOST, help="TCP trigger server host (tcp mode only)")
    parser.add_argument("--port", type=int, default=DEFAULT_TCP_PORT, help="TCP trigger server port (tcp mode only)")
    args = parser.parse_args()

    # If the user only cares about a trigger count, don't make them also work
    # out a matching --hours value -- fall back to the 8h ceiling so time
    # isn't the thing that stops the run first.
    if args.hours is None:
        max_hours = DEFAULT_MAX_HOURS if args.max_triggers is None else 8.0
    else:
        max_hours = args.hours

    stop_event = threading.Event()

    try:
        if args.mode == "tcp":
            run_q_trigger_tcp(
                stop_event,
                host=args.host,
                port=args.port,
                delay_s=args.delay,
                interval_s=args.interval,
                max_hours=max_hours,
                max_triggers=args.max_triggers,
            )
        else:
            run_q_trigger(
                stop_event,
                delay_s=args.delay,
                interval_s=args.interval,
                max_hours=max_hours,
                max_triggers=args.max_triggers,
            )
    except KeyboardInterrupt:
        stop_event.set()
        log.info("Automated trigger interrupted")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    main()