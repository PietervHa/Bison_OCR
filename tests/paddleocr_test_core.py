import os
import json
from pathlib import Path

import cv2

# Keep these flags set before importing PaddleOCR.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from paddleocr import PaddleOCR
import pytesseract

from backend.core.config_loader import cfg

# Create tests/results folder if it doesn't exist
_tests_root = Path(__file__).resolve().parent
output_dir = _tests_root / "results"
output_dir.mkdir(exist_ok=True)

def _init_paddle():
    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _configure_tesseract():
    tesseract_path = cfg.get("ocr", {}).get("tesseract_path", "")
    if tesseract_path and Path(tesseract_path).exists():
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


def _save_tesseract_outputs(image_path: Path, out_dir: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    detections = []
    for i in range(len(data["text"])):
        text = (data["text"][i] or "").strip()
        conf_raw = data["conf"][i]
        try:
            confidence = float(conf_raw)
        except (TypeError, ValueError):
            confidence = -1.0

        if not text or confidence < 0:
            continue

        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])

        detections.append(
            {
                "text": text,
                "confidence": confidence,
                "bbox": [x, y, w, h],
            }
        )
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            text,
            (x, max(0, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    stem = image_path.stem
    annotated_path = out_dir / f"{stem}_annotated.png"
    json_path = out_dir / f"{stem}.json"
    cv2.imwrite(str(annotated_path), image)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "engine": "tesseract",
                "image": image_path.name,
                "detections": detections,
            },
            f,
            indent=2,
        )


_configure_tesseract()
ocr = None
try:
    ocr = _init_paddle()
except Exception as e:
    print(f"[WARN] PaddleOCR initialization failed, using Tesseract fallback. Error: {e}")

# Scan and process all images in tests/images folder
test_images_dir = _tests_root / "images"
image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

for image_path in sorted(test_images_dir.iterdir()):
    if image_path.suffix.lower() in image_extensions:
        print(f"\nProcessing: {image_path.name}")

        try:
            if ocr is not None:
                result = ocr.predict(input=str(image_path))
                for res in result:
                    res.print()
                    res.save_to_img(str(output_dir))
                    res.save_to_json(str(output_dir))
                print(f"[SUCCESS] Paddle results saved for: {image_path.name}")
            else:
                _save_tesseract_outputs(image_path, output_dir)
                print(f"[SUCCESS] Tesseract results saved for: {image_path.name}")
        except Exception as e:
            # If Paddle inference fails at runtime, fallback per image.
            if ocr is not None:
                print(f"[WARN] Paddle failed for {image_path.name}, falling back to Tesseract. Error: {e}")
                try:
                    _save_tesseract_outputs(image_path, output_dir)
                    print(f"[SUCCESS] Tesseract results saved for: {image_path.name}")
                    continue
                except Exception as fallback_error:
                    print(f"[ERROR] Tesseract fallback failed for {image_path.name}: {fallback_error}")
                    continue

            print(f"[ERROR] Error processing {image_path.name}: {e}")
            continue

print("\nProcessing complete!")


