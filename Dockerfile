# Worker serverless de WhisperX (transcripción + alineación de letra) en GPU.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# torch CUDA (cu121) — whisperx usa faster-whisper (ctranslate2) + wav2vec2 (torch).
RUN pip3 install --no-cache-dir torch==2.2.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir whisperx runpod

# Pre-descarga el modelo Whisper para que el cold start no lo baje (int8 en CPU solo
# para cachear los pesos; en runtime se carga en GPU con float16).
RUN python3 -c "import whisperx; whisperx.load_model('large-v3','cpu',compute_type='int8')" || true

COPY handler.py /handler.py
CMD ["python3", "-u", "/handler.py"]
