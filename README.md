# whisperx-serverless

Worker de RunPod Serverless que transcribe y **sincroniza la letra** de una canción con
**WhisperX** (GPU). Lo consume Fototeca para generar karaoke automático (sin archivos LRC).

## Contrato

Transcribir (por defecto):
```
input:  { "audio_url": "<url R2>", "language": "es" }   # language opcional (autodetecta)
output: { "language": "es", "segments": [ { "start", "end", "text", "words":[…] } ] }
```

Traducir la letra (MADLAD-400):
```
input:  { "task": "translate", "lines": ["verso 1", …], "targets": ["en","pt"] }
output: { "translations": { "en": ["verse 1", …], "pt": [ … ] } }
```
Devuelve exactamente las mismas líneas que entran: el karaoke reutiliza los tiempos
del original, así que perder o juntar una línea corre el subtítulo contra la música.

La salida es JSON chico → inline (no usa R2 de salida).

## Deploy (igual que demucs-serverless)
1. Push este repo a GitHub.
2. RunPod → Serverless → New Endpoint → **Deploy from a GitHub repository** → este repo.
3. GPU: **24GB recomendado** (Whisper large-v3 ~10GB + MADLAD-3b fp16 ~6GB). En 16GB
   funciona: al cargar MADLAD suelta Whisper y lo recarga cuando vuelva a transcribir.
4. **Container disk ≥45GB** — los pesos de MADLAD van dentro de la imagen (~12GB) para
   que el cold start no los baje. Con los 20GB de antes el build no cabe.
5. Min workers 0, idle 5s, execution timeout 600s.
6. Copia el **Endpoint ID** → Fototeca `.env`: `WHISPERX_ENDPOINT_ID` (usa el mismo `RUNPOD_API_KEY`).

Nota: si 16GB escasea, baja a `WHISPER_MODEL=medium` (menos VRAM, casi igual de bueno para letra).
