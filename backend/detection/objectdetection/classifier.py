"""
Local Image Classifier

Provides a wrapper for Hugging Face-based image classification models running 
locally on CPU or GPU.
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING, Dict
from PIL import Image
from backend.utils.logger import get_logger
if TYPE_CHECKING:
    import torch
    from transformers import AutoImageProcessor, AutoModelForImageClassification

logger = get_logger(__name__)

class ImageClassifier:
    """Image classifier wrapper backed by a local Hugging Face model path."""

    def __init__(self, model_path: str, device: str = None) -> None:
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None

        self._load()
        if self.is_loaded():
            logger.info("Classifier loaded: model_path=%s device=%s", self.model_path, self.device)
        else:
            logger.warning("Classifier remains unloaded after load attempt: model_path=%s", self.model_path)

    def _load(self) -> None:
        """
        Internal method to load the model and processor from the specified path.
        """
        model_dir = Path(self.model_path)
        if not model_dir.is_absolute():
            model_dir = (Path(__file__).resolve().parents[3] / model_dir).resolve()

        if not model_dir.exists():
            logger.warning("Classifier model path not found: %s. Classifier will stay unloaded.", model_dir)
            self.model = None
            self.processor = None
            return

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification

            self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model_dir_str = str(model_dir)
            self.processor = AutoImageProcessor.from_pretrained(model_dir_str)
            # use_safetensors=True refuses to fall back to a pickled pytorch_model.bin,
            # which can execute arbitrary code on load. Only accept safetensors weights.
            self.model = AutoModelForImageClassification.from_pretrained(
                model_dir_str, use_safetensors=True
            )
            self.model.to(self.device)
            self.model.eval()
            self.model_path = model_dir_str
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Unable to load classifier assets from '{model_dir}'. "
                "Ensure required model files are present (for example config and weights)."
            ) from exc
        except OSError as exc:
            # transformers raises OSError (not FileNotFoundError) when no
            # safetensors weights are found in an otherwise valid model dir.
            raise RuntimeError(
                f"Classifier at '{model_dir}' has no safetensors weights. "
                "Only .safetensors checkpoints are accepted (pickled .bin files are "
                "rejected for security reasons); re-export the model as safetensors."
            ) from exc

    def predict(self, pil_image: Image.Image) -> Dict[str, object]:
        """Run inference on a single RGB PIL image."""
        if self.model is None or self.processor is None:
            logger.warning("Classifier predict called while model is not loaded.")
            return {"label": "unavailable", "confidence": 0.0, "all_scores": {}}

        if not isinstance(pil_image, Image.Image):
            raise TypeError("pil_image must be a PIL.Image.Image instance")
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        import torch

        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0]

        top_index = int(torch.argmax(probabilities).item())
        id2label = getattr(self.model.config, "id2label", {}) or {}
        top_label = id2label.get(top_index, str(top_index))

        all_scores = {}
        for class_index, score in enumerate(probabilities.detach().cpu().tolist()):
            label = id2label.get(class_index, str(class_index))
            all_scores[label] = float(score)

        confidence = round(float(probabilities[top_index].item()), 3)
        return {
            "label": top_label,
            "confidence": confidence,
            "all_scores": all_scores,
        }

    def warmup(self) -> None:
        """Runs one throwaway inference so the native-runtime cold-start cost
        (thread pool spin-up, oneDNN/MKL kernel selection) is paid at startup
        instead of during the first real inspection cycle."""
        if self.model is None or self.processor is None:
            return
        try:
            self.predict(Image.new("RGB", (224, 224)))
            logger.debug("Classifier warm-up inference completed.")
        except Exception as exc:
            logger.warning("Classifier warm-up inference failed (non-fatal): %s", exc)

    def is_loaded(self) -> bool:
        """
        Checks if the model and processor have been successfully loaded.
        """
        return self.model is not None