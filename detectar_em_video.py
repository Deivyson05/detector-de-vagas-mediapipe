import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import argparse
import time
import os

# ===========================================================================
# ✏️  CONFIGURAÇÃO
# ===========================================================================

MODEL_FILE      = "exported_model/model.tflite"
SCORE_THRESHOLD = 0.25
MAX_RESULTS     = 10

BOX_COLOR   = (0, 255, 0)
TEXT_COLOR  = (0, 255, 0)
STATS_COLOR = (0, 200, 255)

# ===========================================================================


def has_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def build_detector(model_path: str) -> vision.ObjectDetector:
    options = vision.ObjectDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        score_threshold=SCORE_THRESHOLD,
        max_results=MAX_RESULTS,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.ObjectDetector.create_from_options(options)


def draw_detections(frame_bgr: np.ndarray, result: vision.ObjectDetectorResult) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    for det in result.detections:
        bb = det.bounding_box
        x1 = max(0, int(bb.origin_x))
        y1 = max(0, int(bb.origin_y))
        x2 = min(w, int(bb.origin_x + bb.width))
        y2 = min(h, int(bb.origin_y + bb.height))

        cat   = det.categories[0]
        label = cat.category_name or f"classe_{cat.index}"
        score = cat.score

        cv2.rectangle(out, (x1, y1), (x2, y2), BOX_COLOR, 2)

        text = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        label_y = max(th + 4, y1 - 4)
        cv2.rectangle(out, (x1, label_y - th - 4), (x1 + tw + 4, label_y + 2), BOX_COLOR, -1)
        cv2.putText(out, text, (x1 + 2, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    return out


def overlay_stats(frame: np.ndarray, fps: float, n_det: int, frame_no: int) -> None:
    lines = [
        f"FPS: {fps:5.1f}",
        f"Deteccoes: {n_det}",
        f"Frame: {frame_no}",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, STATS_COLOR, 2)
        y += 26


def make_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Tenta codecs em ordem até um funcionar."""
    codecs = ["avc1", "mp4v", "XVID"]
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        w = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if w.isOpened():
            print(f"    Codec utilizado: {codec}")
            return w
        w.release()
    raise RuntimeError(
        "Nenhum codec disponível (avc1/mp4v/XVID). "
        "Instale o ffmpeg ou tente salvar como .avi"
    )


def process_video(source, output_path, model_path: str) -> None:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir: {source}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    show_window = has_display()

    print(f"\n=== Fonte: {source} ===")
    print(f"    Resolução : {width}x{height}  |  FPS fonte: {fps_src:.1f}")
    print(f"    Total frames: {total if total > 0 else 'ao vivo (webcam)'}")
    if show_window:
        print("    Pressione  Q  para sair.\n")
    else:
        print("    Modo headless (sem display). Pressione Ctrl+C para interromper.\n")

    writer = None
    if output_path:
        writer = make_writer(output_path, fps_src, width, height)
        print(f"    Salvando em: {output_path}\n")

    frame_no = 0
    fps_disp = 0.0
    t_prev   = time.perf_counter()
    log_step = max(1, int(fps_src * 5))  # progresso a cada ~5 s

    try:
        with build_detector(model_path) as detector:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break

                img_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                result   = detector.detect(mp_image)

                out_frame = draw_detections(frame_bgr, result)

                t_now    = time.perf_counter()
                fps_disp = 0.9 * fps_disp + 0.1 * (1.0 / max(t_now - t_prev, 1e-9))
                t_prev   = t_now

                overlay_stats(out_frame, fps_disp, len(result.detections), frame_no)

                if writer:
                    writer.write(out_frame)

                if frame_no % log_step == 0:
                    pct = f"{frame_no/total*100:.1f}%" if total > 0 else f"{frame_no} frames"
                    print(f"  [{pct}] frame {frame_no}  FPS={fps_disp:.1f}  det={len(result.detections)}")

                frame_no += 1

                if show_window:
                    cv2.imshow("Deteccao em Tempo Real  [Q para sair]", out_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        print("\nInterrompido pelo usuário.")
                        break

    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário (Ctrl+C).")
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"\nVídeo salvo em: {output_path}")
        if show_window:
            cv2.destroyAllWindows()
        print(f"Frames processados: {frame_no}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global SCORE_THRESHOLD, MAX_RESULTS

    parser = argparse.ArgumentParser(
        description="Detecção de objetos em vídeo usando MediaPipe + TFLite"
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--video", "-v", metavar="ARQUIVO",
                               help="Arquivo de vídeo (mp4, avi, mov…)")
    source_group.add_argument("--webcam", "-w", metavar="ID", type=int,
                               nargs="?", const=0,
                               help="Índice da webcam (padrão: 0)")

    parser.add_argument("--output", "-o", metavar="ARQUIVO", default=None,
                        help="Salvar vídeo anotado (ex: resultado.mp4)")
    parser.add_argument("--model", "-m", metavar="ARQUIVO", default=MODEL_FILE,
                        help=f"Caminho do .tflite (padrão: {MODEL_FILE})")
    parser.add_argument("--threshold", "-t", type=float, default=SCORE_THRESHOLD,
                        help=f"Score mínimo (padrão: {SCORE_THRESHOLD})")
    parser.add_argument("--max-results", "-n", type=int, default=MAX_RESULTS,
                        help=f"Máximo de detecções por frame (padrão: {MAX_RESULTS})")

    args = parser.parse_args()

    SCORE_THRESHOLD = args.threshold
    MAX_RESULTS     = args.max_results

    source = args.video if args.video is not None else args.webcam
    process_video(source, args.output, args.model)


if __name__ == "__main__":
    main()