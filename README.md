# 🅿️ Mini Estacionamento Inteligente
 
Projeto acadêmico de uma maquete de estacionamento inteligente que utiliza **visão computacional** para detectar e contabilizar veículos em tempo real, integrada a um sistema físico controlado por **Arduino**.
 
---
 
## 📌 Visão Geral
 
O sistema utiliza um modelo de IA treinado para identificar carros por meio de uma câmera. Ao detectar um veículo entrando ou saindo da maquete, o modelo atualiza a contagem de vagas disponíveis e se comunica com o Arduino, que controla os elementos físicos do estacionamento (cancela, LEDs indicadores, etc.).
 
---
 
## 🧠 Modelo de IA
 
| Item | Detalhe |
|---|---|
| Framework de treinamento | [MediaPipe Model Maker](https://ai.google.dev/edge/mediapipe/solutions/model_maker) |
| Arquitetura base | EfficientDet-Lite0 |
| Tarefa | Detecção de Objetos (Object Detection) |
| Classe detectada | `car` |
 
### Por que EfficientDet-Lite0?
 
O EfficientDet-Lite0 é a variante mais leve da família EfficientDet, otimizada para dispositivos com recursos limitados (edge devices). Isso o torna ideal para rodar em hardware embarcado ou em conjunto com um microcontrolador como o Arduino, mantendo uma boa relação entre velocidade de inferência e precisão.
 
---
 
## ⚙️ Tecnologias Utilizadas
 
- **Python 3.x**
- **MediaPipe Model Maker** — treinamento do modelo de detecção de objetos
- **TensorFlow Lite** — inferência do modelo exportado
- **OpenCV** — captura e processamento de vídeo em tempo real
- **Arduino** — controle físico da maquete (servo motor, LEDs, display LCD)
- **PySerial** — comunicação entre Python e Arduino via porta serial
---
 

## 🔌 Funcionamento do Sistema
 
```
┌─────────────┐     frame      ┌──────────────────┐
│   Câmera    │ ─────────────► │  Modelo de IA    │
└─────────────┘                │ EfficientDet-Lite0│
                               └────────┬─────────┘
                                        │ detecção de carro
                                        ▼
                               ┌──────────────────┐
                               │  Lógica de       │
                               │  Contagem        │
                               └────────┬─────────┘
                                        │ Serial (PySerial)
                                        ▼
                               ┌──────────────────┐
                               │    Arduino       │
                               │  cancela / LEDs  │
                               └──────────────────┘
```
 
1. A câmera captura frames em tempo real.
2. O modelo detecta a presença de um carro na entrada/saída.
3. A lógica de contagem atualiza o número de vagas disponíveis.
4. O Arduino recebe o comando via serial e aciona os atuadores físicos da maquete.
---
 
## 📊 Treinamento — Detalhes
 
O treinamento foi realizado com o **MediaPipe Model Maker**, que simplifica o processo de fine-tuning sobre modelos pré-treinados para detecção de objetos.
 
```python
from mediapipe_model_maker import object_detector
 
# Carregando o dataset
train_data = object_detector.Dataset.from_pascal_voc_folder('dataset/train')
val_data   = object_detector.Dataset.from_pascal_voc_folder('dataset/validation')
 
# Configuração do modelo
spec = object_detector.SupportedModels.EFFICIENTDET_LITE0
 
hparams = object_detector.HParams(export_dir='model/')
options = object_detector.ObjectDetectorOptions(
    supported_model=spec,
    hparams=hparams
)
 
# Treinamento
model = object_detector.ObjectDetector.create(
    train_data=train_data,
    validation_data=val_data,
    options=options
)
 
# Exportação para TFLite
model.export_model()
```
 
---
 
## 👥 Equipe
 
| Nome | Matrícula |
|---|---|
| Deivyson Ricardo Silva dos Santos | 855214 |
| Iwerson Guilherme da Silva Souza | 855213 |
| Ingrid Beatriz Silva | 855232 |
| Júlia Muniz Cavalheiro de Oliveira | 855158 |
| Vinicius de Almeida da Silva | 855166 |
 
---
 
## 📄 Licença
 
Este projeto foi desenvolvido para fins acadêmicos.
