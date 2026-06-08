import cv2
import tensorflow as tf
import numpy as np

SCORE_THRESHOLD = 0.3
INPUT_SIZE = 256

# ── Carrega âncoras reais ──
anchors = np.load("anchors.npy")  # (12276, 4) [cx, cy, w, h]
print(f"Âncoras: {anchors.shape}")

def decode_boxes(raw_boxes, anchors):
    """
    Scale factors = 1.0 conforme metadata.
    Ordem de saída: [xmin, ymin, xmax, ymax] conforme index=[1,0,3,2].
    """
    cx = raw_boxes[:, 0] * anchors[:, 2] + anchors[:, 0]
    cy = raw_boxes[:, 1] * anchors[:, 3] + anchors[:, 1]
    w  = np.exp(raw_boxes[:, 2]) * anchors[:, 2]
    h  = np.exp(raw_boxes[:, 3]) * anchors[:, 3]

    xmin = cx - w / 2
    ymin = cy - h / 2
    xmax = cx + w / 2
    ymax = cy + h / 2

    return np.stack([xmin, ymin, xmax, ymax], axis=-1)

# ── Inferência ──
interpreter = tf.lite.Interpreter(model_path="exported_model/model.tflite")
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

img     = cv2.imread("carro3.jpg")
h0, w0  = img.shape[:2]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_in  = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE))

# Normalização correta: (pixel - 127.5) / 127.5 → range [-1, 1]
input_data = (np.expand_dims(img_in, axis=0).astype(np.float32) - 127.5) / 127.5

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

raw_boxes  = interpreter.get_tensor(output_details[0]['index'])[0]  # (12276, 4)
scores_sig = interpreter.get_tensor(output_details[1]['index'])[0]  # (12276, 2)

scores_obj = scores_sig[:, 1]  # classe 1 = carro
print(f"Top 5 scores: {np.sort(scores_obj)[::-1][:5]}")

# Decode com âncoras reais
decoded = decode_boxes(raw_boxes, anchors)  # (12276, 4) [xmin,ymin,xmax,ymax]

# Filtra pelo threshold
mask     = scores_obj >= SCORE_THRESHOLD
boxes_f  = decoded[mask]
scores_f = scores_obj[mask]
print(f"Detecções acima de {SCORE_THRESHOLD}: {len(boxes_f)}")

if len(boxes_f) > 0:
    # NMS — cv2 espera [x, y, w, h]
    bboxes_cv = []
    for b in boxes_f:
        xmin, ymin, xmax, ymax = b
        bboxes_cv.append([float(xmin), float(ymin),
                          float(xmax - xmin), float(ymax - ymin)])

    indices = cv2.dnn.NMSBoxes(
        bboxes=bboxes_cv,
        scores=scores_f.tolist(),
        score_threshold=SCORE_THRESHOLD,
        nms_threshold=0.1
    )

    img_draw = img.copy()
    for i in indices.flatten():
        xmin, ymin, xmax, ymax = boxes_f[i]
        x1 = int(np.clip(xmin, 0, 1) * w0)
        y1 = int(np.clip(ymin, 0, 1) * h0)
        x2 = int(np.clip(xmax, 0, 1) * w0)
        y2 = int(np.clip(ymax, 0, 1) * h0)
        score = scores_f[i]
        print(f"  Score: {score:.3f} | ({x1},{y1}) → ({x2},{y2})")
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_draw, f"{score:.2f}", (x1, max(10, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imwrite("resultado.jpg", img_draw)
    print("Salvo em resultado.jpg")
else:
    print("Nenhuma detecção. Reduza SCORE_THRESHOLD.")