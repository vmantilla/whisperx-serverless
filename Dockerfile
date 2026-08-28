# Worker serverless de WhisperX (transcripción + alineación de letra) en GPU.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# torch CUDA (cu121) — whisperx usa faster-whisper (ctranslate2) + wav2vec2 (torch).
RUN pip3 install --no-cache-dir torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir whisperx runpod
# MADLAD-400 corre con transformers (T5). sentencepiece es su tokenizador.
RUN pip3 install --no-cache-dir "transformers>=4.40" sentencepiece accelerate

# Pre-descarga el modelo Whisper para que el cold start no lo baje (int8 en CPU solo
# para cachear los pesos; en runtime se carga en GPU con float16).
RUN python3 -c "import whisperx; whisperx.load_model('large-v3','cpu',compute_type='int8')" || true

# Pesos de MADLAD dentro de la imagen: bajarlos en cada cold start serían minutos
# de espera por canción. Engorda la imagen ~12 GB, pero RunPod la cachea en el worker.
ARG MADLAD_MODEL=google/madlad400-3b-mt
ENV MADLAD_MODEL=${MADLAD_MODEL}
RUN python3 -c "\
from transformers import AutoTokenizer, T5ForConditionalGeneration; \
import os; m=os.environ['MADLAD_MODEL']; \
AutoTokenizer.from_pretrained(m); T5ForConditionalGeneration.from_pretrained(m)"

COPY handler.py /handler.py
CMD ["python3", "-u", "/handler.py"]

# Build: 2026-08-28T05:14:10Z
