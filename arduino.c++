#include <Servo.h>

Servo cancela;

const int servoPin = 9;
const int botaoSaida = 2;

const int ledVerde = 6;
const int ledVermelho = 7;

bool ultimoEstadoBotao = HIGH;

void setup() {

  Serial.begin(9600);

  pinMode(botaoSaida, INPUT_PULLUP);

  pinMode(ledVerde, OUTPUT);
  pinMode(ledVermelho, OUTPUT);

  cancela.attach(servoPin);

  cancela.write(0);

  digitalWrite(ledVerde, HIGH);
}

void loop() {

  if (Serial.available()) {

    char comando = Serial.read();

    // ABRIR CANCELA

    if (comando == 'A') {

      digitalWrite(ledVerde, HIGH);
      digitalWrite(ledVermelho, LOW);

      cancela.write(90);

      delay(3000);

      cancela.write(0);
    }

    // LOTADO

    if (comando == 'F') {

      digitalWrite(ledVerde, LOW);
      digitalWrite(ledVermelho, HIGH);
    }

  }

  bool estadoAtual = digitalRead(botaoSaida);

  if (
      ultimoEstadoBotao == HIGH &&
      estadoAtual == LOW
     ) {

    Serial.println("SAIU");

    delay(300);
  }

  ultimoEstadoBotao = estadoAtual;
}