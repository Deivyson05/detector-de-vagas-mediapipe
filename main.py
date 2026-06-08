from mediapipe_model_maker import object_detector

print("Carregando datasets...")

train_data = object_detector.Dataset.from_coco_folder(
    "dataset/train"
)

validation_data = object_detector.Dataset.from_coco_folder(
    "dataset/validation"
)

print("Configurando treinamento...")

spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG

hparams = object_detector.HParams(
    export_dir="exported_model"
)

options = object_detector.ObjectDetectorOptions(
    supported_model=spec,
    hparams=hparams
)

print("Iniciando treinamento...")

model = object_detector.ObjectDetector.create(
    train_data=train_data,
    validation_data=validation_data,
    options=options
)

print("Exportando modelo...")

model.export_model()

print("Concluído!")