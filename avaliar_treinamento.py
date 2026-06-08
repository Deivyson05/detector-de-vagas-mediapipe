"""
Avalia um modelo .tflite de detecção de objetos contra anotações COCO.
Calcula: Precision, Recall, F1, mAP@0.5 e mAP@0.5:0.95.

Não retreina nem modifica o modelo — só lê e avalia.
"""

import json
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pathlib import Path
from collections import defaultdict

# ===========================================================================
# ✏️  CONFIGURAÇÃO
# ===========================================================================

MODEL_FILE      = "exported_model/model.tflite"  # ← seu .tflite
COCO_JSON       = "dataset/validation/labels.json"  # ← arquivo COCO de validação
IMAGES_DIR      = "dataset/validation/images"                    # ← pasta com as imagens de validação
SCORE_THRESHOLD = 0.25   # baixo para capturar tudo e calcular curva precision/recall
MAX_RESULTS     = 100
IOU_THRESHOLD   = 0.5    # IoU mínimo para considerar detecção correta

# ===========================================================================


def build_detector(model_path: str) -> vision.ObjectDetector:
    options = vision.ObjectDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        score_threshold=SCORE_THRESHOLD,
        max_results=MAX_RESULTS,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.ObjectDetector.create_from_options(options)


def iou(boxA, boxB) -> float:
    """Calcula IoU entre dois boxes [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / (areaA + areaB - inter)


def load_coco(json_path: str):
    """
    Retorna:
      images      : {image_id: {file_name, width, height}}
      gt_by_image : {image_id: [{box:[x1,y1,x2,y2], category_id, category_name}]}
      categories  : {category_id: name}
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data["categories"]}

    images = {img["id"]: img for img in data["images"]}

    gt_by_image = defaultdict(list)
    for ann in data["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        x, y, w, h = ann["bbox"]
        gt_by_image[ann["image_id"]].append({
            "box": [x, y, x + w, y + h],
            "category_id": ann["category_id"],
            "category_name": categories[ann["category_id"]],
        })

    return images, gt_by_image, categories


def compute_ap(precisions, recalls) -> float:
    """AP pela interpolação de 11 pontos (VOC style)."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = [p for p, r in zip(precisions, recalls) if r >= t]
        ap += max(p) if p else 0.0
    return ap / 11


def evaluate(model_path, coco_json, images_dir, iou_threshold=0.5):
    images_dir = Path(images_dir)
    images_meta, gt_by_image, categories = load_coco(coco_json)

    # Agrupa por categoria para calcular AP por classe
    # all_preds[cat_name] = [(score, is_tp), ...]
    all_preds   = defaultdict(list)
    gt_counts   = defaultdict(int)   # total de GTs por categoria

    total_images = len(images_meta)
    print(f"\n=== Avaliando {total_images} imagens ===")
    print(f"  Modelo     : {model_path}")
    print(f"  Anotações  : {coco_json}")
    print(f"  IoU mínimo : {iou_threshold}\n")

    with build_detector(model_path) as detector:
        for idx, (img_id, img_info) in enumerate(images_meta.items(), 1):
            img_path = images_dir / img_info["file_name"]
            if not img_path.exists():
                # tenta só pelo nome base caso o path COCO seja relativo
                img_path = images_dir / Path(img_info["file_name"]).name
            if not img_path.exists():
                print(f"  [AVISO] Imagem não encontrada: {img_info['file_name']}")
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            h, w = img_bgr.shape[:2]
            img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result   = detector.detect(mp_image)

            gts = gt_by_image.get(img_id, [])
            for gt in gts:
                gt_counts[gt["category_name"]] += 1

            # Marca quais GTs já foram usados (evita contar dois TPs pro mesmo GT)
            matched_gts = set()

            # Ordena predições por score decrescente
            preds = sorted(result.detections, key=lambda d: d.categories[0].score, reverse=True)

            for det in preds:
                cat   = det.categories[0]
                label = cat.category_name or f"classe_{cat.index}"
                score = cat.score
                bb    = det.bounding_box

                # Converte box normalizado → pixels
                pred_box = [
                    bb.origin_x, bb.origin_y,
                    bb.origin_x + bb.width,
                    bb.origin_y + bb.height,
                ]

                # Procura GT correspondente com maior IoU
                best_iou  = 0.0
                best_gt_i = -1
                for gi, gt in enumerate(gts):
                    if gt["category_name"] != label:
                        continue
                    if gi in matched_gts:
                        continue
                    i = iou(pred_box, gt["box"])
                    if i > best_iou:
                        best_iou  = i
                        best_gt_i = gi

                is_tp = best_iou >= iou_threshold and best_gt_i >= 0
                if is_tp:
                    matched_gts.add(best_gt_i)

                all_preds[label].append((score, int(is_tp)))

            if idx % 50 == 0 or idx == total_images:
                print(f"  Progresso: {idx}/{total_images} imagens")

    # ── Calcula métricas por categoria ──
    print("\n" + "=" * 65)
    print(f"{'Categoria':<25} {'GT':>5} {'Det':>5} {'Prec':>7} {'Recall':>7} {'AP@.5':>7}")
    print("=" * 65)

    aps = []
    for cat_name in sorted(gt_counts.keys()):
        n_gt   = gt_counts[cat_name]
        preds_c = sorted(all_preds[cat_name], key=lambda x: x[0], reverse=True)
        n_det  = len(preds_c)

        if n_det == 0:
            print(f"  {cat_name:<23} {n_gt:>5} {0:>5} {'—':>7} {'—':>7} {'—':>7}")
            aps.append(0.0)
            continue

        tp_cum = np.cumsum([p[1] for p in preds_c])
        fp_cum = np.cumsum([1 - p[1] for p in preds_c])

        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
        recalls    = tp_cum / (n_gt + 1e-9)

        ap  = compute_ap(precisions.tolist(), recalls.tolist())
        p   = precisions[-1]
        r   = recalls[-1]

        aps.append(ap)
        print(f"  {cat_name:<23} {n_gt:>5} {n_det:>5} {p:>7.3f} {r:>7.3f} {ap:>7.3f}")

    map50 = float(np.mean(aps)) if aps else 0.0

    print("=" * 65)
    print(f"  {'mAP@0.5':<23} {' ':>5} {' ':>5} {' ':>7} {' ':>7} {map50:>7.3f}")
    print("=" * 65)
    print(f"\n  Total de categorias avaliadas : {len(gt_counts)}")
    print(f"  mAP@0.5                       : {map50:.4f}")


if __name__ == "__main__":
    evaluate(
        model_path    = MODEL_FILE,
        coco_json     = COCO_JSON,
        images_dir    = IMAGES_DIR,
        iou_threshold = IOU_THRESHOLD,
    )