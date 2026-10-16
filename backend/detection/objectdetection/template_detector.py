"""
Template Matching Detector

Provides the TemplateDetector class which uses OpenCV template matching and 
Structural Similarity Index (SSIM) to verify the presence of objects based on 
reference images.
"""

import time
import cv2
import numpy as np
from pathlib import Path
from backend.utils.logger import get_logger

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    def ssim_fn(a, b, data_range):
        return float(cv2.quality.QualitySSIM_compute(a, b)[0][0])


class TemplateDetector:
    """
    Detection engine that compares input frames against a set of reference 
    images using SSIM and template matching.
    """
    def __init__(self, reference_paths: list[str], match_threshold: float = 0.6, display_label: str = ""):
        self.logger = get_logger(__name__)
        self.match_threshold = float(match_threshold)
        self.display_label = str(display_label).strip()
        self.references: list[dict] = []

        for reference_path in reference_paths:
            path_obj = Path(reference_path)
            if not path_obj.exists():
                self.logger.warning("Template reference not found, skipping: %s", reference_path)
                continue

            reference_image = cv2.imread(str(path_obj), cv2.IMREAD_GRAYSCALE)
            if reference_image is None:
                self.logger.warning("Failed to load template reference, skipping: %s", reference_path)
                continue

            # In __init__, pre-compute
            prepared = self._prepare_gray_for_ssim(reference_image)
            self.references.append({"image": reference_image, "prepared": prepared, "path": str(path_obj)})

        self.logger.info("TemplateDetector loaded %d reference image(s)", len(self.references))

        if not self.references:
            raise RuntimeError("No template references loaded successfully.")

    @staticmethod
    def _prepare_gray_for_ssim(image: np.ndarray) -> np.ndarray:
        """Return a 2D uint8 grayscale image suitable for SSIM."""
        if image is None:
            raise ValueError("Image is None")

        img = np.asarray(image)
        if img.ndim == 3 and img.shape[2] == 1:
            img = img[:, :, 0]
        elif img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim != 2:
            raise ValueError(f"Unsupported image shape for SSIM: {img.shape}")

        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            img = img.astype(np.uint8)

        return np.ascontiguousarray(img)

    def detect(self, frame: np.ndarray) -> dict:
        """
        Runs template matching on the input frame and returns the best match result.
        """
        start = time.perf_counter()

        try:
            if frame is None:
                raise ValueError("Input frame is None")

            if frame.ndim == 2:
                frame_gray = frame
            else:
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            candidates = []
            for reference in self.references:
                ref_gray = reference["prepared"]
                ref_h, ref_w = ref_gray.shape[:2]
                frm_h, frm_w = frame_gray.shape[:2]

                if frm_h < ref_h or frm_w < ref_w:
                    self.logger.warning(
                        "Frame is smaller than reference %s (%sx%s < %sx%s), skipping",
                        reference["path"],
                        frm_w,
                        frm_h,
                        ref_w,
                        ref_h,
                    )
                    continue

                match = cv2.matchTemplate(frame_gray, ref_gray, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(match)
                if max_val < self.match_threshold * 0.7:
                    continue  # skip SSIM — no chance of passing

                x, y = max_loc
                crop = frame_gray[y : y + ref_h, x : x + ref_w]
                if crop.shape[:2] != ref_gray.shape[:2]:
                    crop = cv2.resize(crop, (ref_gray.shape[1], ref_gray.shape[0]),
                                      interpolation=cv2.INTER_LINEAR)
                crop = self._prepare_gray_for_ssim(crop)

                if crop.shape != ref_gray.shape:
                    self.logger.warning(
                        "Skipping SSIM for %s due to shape mismatch crop=%s ref=%s",
                        reference["path"],
                        crop.shape,
                        ref_gray.shape,
                    )
                    continue

                try:
                    ssim_score = float(ssim_fn(crop, ref_gray, data_range=255))
                except Exception as exc:
                    self.logger.warning(
                        "Skipping SSIM for %s: %s (crop=%s ref=%s)",
                        reference["path"],
                        exc,
                        crop.shape,
                        ref_gray.shape,
                    )
                    continue
                candidates.append(
                    {
                        "match_score": float(max_val),
                        "ssim_score": ssim_score,
                        "location": (int(x), int(y)),
                        "path": reference["path"],
                    }
                )

            if not candidates:
                raise RuntimeError("No valid template matches computed.")

            best = max(candidates, key=lambda item: item["ssim_score"])
            best_ssim = float(best["ssim_score"])
            best_match_score = float(best["match_score"])
            elapsed_ms = float((time.perf_counter() - start) * 1000.0)

            return {
                "status": "OK" if best_ssim >= self.match_threshold else "NOK",
                "confidence": round(best_ssim, 3),
                "label": self.display_label,
                "detections": [
                    {
                        "label": self.display_label,
                        "confidence": round(best_ssim, 3),
                        "text": "template_match",
                    }
                ],
                "match_score": round(best_match_score, 3),
                "best_reference": best["path"],
                "processing_time_ms": elapsed_ms,
            }
        except Exception as exc:
            self.logger.error("TemplateDetector.detect() failed: %s", exc, exc_info=True)
            elapsed_ms = float((time.perf_counter() - start) * 1000.0)
            return {
                "status": "NOK",
                "confidence": 0.0,
                "detections": [],
                "error": str(exc),
                "processing_time_ms": elapsed_ms,
            }

    def is_loaded(self) -> bool:
        """
        Checks if any reference images were successfully loaded.
        """
        return len(self.references) > 0


