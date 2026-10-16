"""
Application State Management

This module defines the AppState class, which holds the shared state for the 
entire vision application, ensuring thread-safe access across camera, vision, and web threads.
"""

import threading
from backend.core.config_loader import cfg

class AppState:
    """
    Thread-safe container for the application's runtime state.
    """
    def __init__(self):
        # The lock keeps camera, vision, and web threads from reading/writing
        # shared state at the same time.
        self.lock = threading.Lock()
        self.latest_result = {
            "detections": [],
            "processing_time_ms": 0,
            "status": "NOK",
        }
        self.counters = {
            "ok": 0,
            "nok": 0,
            "total": 0,
        }
        self.confidence_threshold = 0.0
        self.maintenance_mode = False
        self.maintenance_session_token = ""
        self.maintenance_session_user = ""
        self.camera_rotation = 0  # steps: 0, 1, 2, 3 (each = 90° clockwise)
        self.set_threshold(float(cfg["confidence_threshold"]))

    def update_result(self, result: dict):
        """
        Updates the latest vision processing result.
        """
        with self.lock:
            self.latest_result = result

    def increment_counter(self, status: str):
        """
        Increments the OK/NOK counters based on the provided status.
        """
        with self.lock:
            self.counters["total"] += 1
            if status == "OK":
                self.counters["ok"] += 1
            elif status == "NOK":
                self.counters["nok"] += 1

    def reset_counters(self):
        """
        Resets all inspection counters to zero.
        """
        with self.lock:
            self.counters = {
                "ok": 0,
                "nok": 0,
                "total": 0,
            }

    def get_snapshot(self) -> dict:
        """
        Returns a point-in-time copy of the core application state.
        """
        with self.lock:
            return {
                "result": dict(self.latest_result),
                "counters": dict(self.counters),
                "maintenance_mode": self.maintenance_mode,
            }

    def get_threshold(self) -> float:
        """
        Gets the current confidence threshold.
        """
        with self.lock:
            return self.confidence_threshold

    def set_threshold(self, value: float):
        """
        Sets the confidence threshold, clamped between 0.0 and 1.0.
        """
        with self.lock:
            self.confidence_threshold = max(0.0, min(1.0, float(value)))

    def get_maintenance_mode(self) -> bool:
        """
        Checks if the application is in maintenance mode.
        """
        with self.lock:
            return self.maintenance_mode

    def set_maintenance_mode(self, value: bool):
        """
        Enables or disables maintenance mode.
        """
        with self.lock:
            self.maintenance_mode = bool(value)

    def get_maintenance_session_token(self) -> str:
        """
        Gets the session token that authorizes maintenance mode in the browser.
        """
        with self.lock:
            return self.maintenance_session_token

    def set_maintenance_session_token(self, value: str):
        """
        Stores or clears the maintenance session token.
        """
        with self.lock:
            self.maintenance_session_token = str(value or "")

    def get_maintenance_session_user(self) -> str:
        """
        Gets the username of whoever is currently authenticated for maintenance mode.
        """
        with self.lock:
            return self.maintenance_session_user

    def set_maintenance_session_user(self, value: str):
        """
        Stores or clears the username tied to the current maintenance session.
        """
        with self.lock:
            self.maintenance_session_user = str(value or "")

    def get_camera_rotation(self) -> int:
        """
        Gets the current camera rotation (0-3 steps of 90°).
        """
        with self.lock:
            return self.camera_rotation

    def rotate_camera(self):
        """
        Increments camera rotation by 90° clockwise.
        """
        with self.lock:
            self.camera_rotation = (self.camera_rotation + 1) % 4
