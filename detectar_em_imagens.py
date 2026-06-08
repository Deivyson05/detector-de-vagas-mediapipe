import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ===========================================================================
# ✏️  CONFIGURAÇÃO
# ===========================================================================

MODEL_FILE      = "exported_model/model.tflite"  # ← seu .tflite diretamente
IMAGE_FILE      = "carros.png"
OUTPUT_FILE     = "resultado.jpg"
SCORE_THRESHOLD = 0.4
MAX_RESULTS     = 10

# ===========================================================================


def build_detector(model_path: str) -> vision.ObjectDetector:
    options = vision.ObjectDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        score_threshold=SCORE_THRESHOLD,
        max_results=MAX_RESULTS,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.ObjectDetector.create_from_options(options)


def draw_detections(img_bgr: np.ndarray, result: vision.ObjectDetectorResult) -> np.ndarray:
    out = img_bgr.copy()
    h, w = out.shape[:2]

    for det in result.detections:
        bb = det.bounding_box
        x1, y1 = bb.origin_x, bb.origin_y
        x2, y2 = x1 + bb.width, y1 + bb.height

        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        cat   = det.categories[0]
        label = cat.category_name or f"classe_{cat.index}"
        score = cat.score

        print(f"  {label:20s}  score={score:.3f}  ({x1},{y1})→({x2},{y2})")

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            out,
            f"{label} {score:.2f}",
            (x1, max(10, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 255, 0), 2,
        )

    return out


def main():
    img_bgr = cv2.imread(IMAGE_FILE)
    if img_bgr is None:
        raise FileNotFoundError(f"Imagem não encontrada: {IMAGE_FILE}")

    img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    print(f"\n=== Detectando em '{IMAGE_FILE}' ===")
    with build_detector(MODEL_FILE) as detector:
        result = detector.detect(mp_image)

    print(f"Detecções encontradas: {len(result.detections)}")

    if result.detections:
        out_img = draw_detections(img_bgr, result)
        cv2.imwrite(OUTPUT_FILE, out_img)
        print(f"\nSalvo em {OUTPUT_FILE}")
    else:
        print("Nenhuma detecção. Tente reduzir SCORE_THRESHOLD.")


if __name__ == "__main__":
    main()