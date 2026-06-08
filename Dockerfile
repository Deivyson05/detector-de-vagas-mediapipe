FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0

RUN pip install --upgrade pip
RUN pip install mediapipe-model-maker

CMD ["bash"]