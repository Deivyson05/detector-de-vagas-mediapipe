"""
Converte um modelo MediaPipe (.tflite + metadata.json) para o formato .task.
Edite a seção CONFIG abaixo com os caminhos dos seus arquivos.
"""

import json
import zipfile
import sys
from pathlib import Path
from datetime import datetime


# ===========================================================================
# ✏️  CONFIGURAÇÃO — edite aqui
# ===========================================================================

CONFIG = {
    # Caminho para o arquivo .tflite exportado pelo MediaPipe
    "model":        "exported_model/model.tflite",

    # Caminho para o metadata.json gerado junto com o .tflite
    # (deixe None se não tiver)
    "metadata":     "exported_model/metadata.json",

    # Onde salvar o .task gerado
    "output":       "model.task",

    # Tipo de tarefa — escolha um:
    #   OBJECT_DETECTOR | IMAGE_CLASSIFIER | IMAGE_SEGMENTER
    #   GESTURE_RECOGNIZER | HAND_LANDMARKER | FACE_LANDMARKER
    #   POSE_LANDMARKER | TEXT_CLASSIFIER | LANGUAGE_DETECTOR
    "task_type":    "OBJECT_DETECTOR",

    # Informações do modelo
    "name":         "Meu Detector Customizado",
    "version":      "1.0.0",
    "description":  "",
    "author":       "",
    "license":      "Apache-2.0",

    # Opções do detector/classificador
    "score_threshold": 0.3,
    "max_results":     10,

    # Arquivos extras para incluir no pacote (lista de caminhos)
    # Ex: ["caminho/vocab.txt", "caminho/anchors.csv"]
    "extra_files":  [],
}

# ===========================================================================


def build_manifest(cfg: dict, has_metadata: bool) -> dict:
    manifest = {
        "schema_version": "1",
        "task_type":      cfg["task_type"],
        "name":           cfg["name"],
        "version":        cfg["version"],
        "description":    cfg["description"],
        "author":         cfg["author"],
        "license":        cfg["license"],
        "created_at":     datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_file":     "model.tflite",
    }

    if has_metadata:
        manifest["metadata_file"] = "metadata.json"

    if cfg["task_type"] == "OBJECT_DETECTOR":
        manifest["object_detection_options"] = {
            "max_results":       cfg["max_results"],
            "score_threshold":   cfg["score_threshold"],
            "category_allowlist": [],
            "category_denylist":  [],
        }
    elif cfg["task_type"] == "IMAGE_CLASSIFIER":
        manifest["classification_options"] = {
            "max_results":     cfg["max_results"],
            "score_threshold": cfg["score_threshold"],
        }

    return manifest


def package_task(cfg: dict):
    tflite_path  = Path(cfg["model"])
    output_path  = Path(cfg["output"])
    metadata_path = Path(cfg["metadata"]) if cfg.get("metadata") else None
    extra_files  = [Path(e) for e in cfg.get("extra_files", [])]

    # Validações
    if not tflite_path.exists():
        sys.exit(f"[ERRO] Modelo não encontrado: {tflite_path}")

    if not output_path.suffix:
        output_path = output_path.with_suffix(".task")

    has_metadata = metadata_path is not None and metadata_path.exists()
    if metadata_path and not has_metadata:
        print(f"[AVISO] metadata.json não encontrado em {metadata_path}, será ignorado.")

    # Se o metadata.json existir, lê descrição de lá (se não preenchida)
    if has_metadata and not cfg["description"]:
        try:
            with open(metadata_path, encoding="utf-8") as f:
                meta = json.load(f)
            cfg["description"] = meta.get("description", "")
        except Exception:
            pass

    manifest = build_manifest(cfg, has_metadata)

    print(f"\n=== Empacotando modelo MediaPipe ===")
    print(f"  Tipo de tarefa : {cfg['task_type']}")
    print(f"  Modelo         : {tflite_path}")
    print(f"  Metadata       : {metadata_path or '(nenhum)'}")
    print(f"  Saída          : {output_path}\n")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        zf.write(tflite_path, "model.tflite")

        if has_metadata:
            zf.write(metadata_path, "metadata.json")

        for ef in extra_files:
            if ef.exists():
                zf.write(ef, ef.name)
            else:
                print(f"[AVISO] Arquivo extra não encontrado, pulando: {ef}")

    size_kb = output_path.stat().st_size / 1024
    print(f"[OK] Arquivo .task gerado: {output_path}  ({size_kb:.1f} KB)")
    print("\nConteúdo do pacote:")
    with zipfile.ZipFile(output_path, "r") as zf:
        for info in zf.infolist():
            print(f"  {info.filename:30s}  {info.file_size / 1024:8.1f} KB")


if __name__ == "__main__":
    package_task(CONFIG)