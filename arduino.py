import cv2
import time
import serial
import numpy as np
import tensorflow as tf

# ==================
# CONFIG
# ==================

CAPACIDADE_MAXIMA = 10
MODEL_PATH = "exported_model/model.tflite"

arduino = serial.Serial("COM3", 9600)
time.sleep(2)

# ==================
# TFLITE
# ==================

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Entrada:")
print(input_details)

print("Saídas:")
for output in output_details:
    print(output)

input_height = input_details[0]["shape"][1]
input_width = input_details[0]["shape"][2]

# ==================
# CAMERA
# ==================

cap = cv2.VideoCapture(0)

vagas_ocupadas = 0
ultima_entrada = 0

COOLDOWN = 3

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    # ==================
    # PREPROCESSAMENTO
    # ==================

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        rgb,
        (input_width, input_height)
    )

    input_data = np.expand_dims(
        resized,
        axis=0
    )

    input_data = input_data.astype(
        np.uint8
    )

    # ==================
    # INFERÊNCIA
    # ==================

    interpreter.set_tensor(
        input_details[0]["index"],
        input_data
    )

    interpreter.invoke()

    # ==================
    # SAÍDAS
    # ==================

    outputs = []

    for output in output_details:

        outputs.append(
            interpreter.get_tensor(
                output["index"]
            )
        )

    # ==================
    # DEBUG
    # ==================

    # MOSTRA O FORMATO DAS SAÍDAS
    # APENAS UMA VEZ

    if len(outputs) > 0:

        print(
            [o.shape for o in outputs]
        )

    # ==================
    # SUA LÓGICA
    # ==================

    carro_detectado = False

    #
    # AQUI VAI DEPENDER DO
    # FORMATO REAL DO MODELO
    #
    # Depois que você rodar,
    # me manda:
    #
    # print(output_details)
    # print([o.shape for o in outputs])
    #
    # e eu monto a leitura
    # correta das bbox.
    #

    if carro_detectado:

        if (
            time.time() - ultima_entrada
            > COOLDOWN
        ):

            ultima_entrada = time.time()

            if vagas_ocupadas < CAPACIDADE_MAXIMA:

                vagas_ocupadas += 1

                arduino.write(b"A")

                print(
                    f"ENTROU {vagas_ocupadas}"
                )

            else:

                arduino.write(b"F")

                print(
                    "LOTADO"
                )

    if arduino.in_waiting:

        msg = (
            arduino.readline()
            .decode()
            .strip()
        )

        if msg == "SAIU":

            vagas_ocupadas = max(
                0,
                vagas_ocupadas - 1
            )

            print(
                f"SAIU {vagas_ocupadas}"
            )

    vagas_livres = (
        CAPACIDADE_MAXIMA
        - vagas_ocupadas
    )

    cv2.putText(
        frame,
        f"Ocupadas: {vagas_ocupadas}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.putText(
        frame,
        f"Livres: {vagas_livres}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    cv2.imshow(
        "Estacionamento",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()