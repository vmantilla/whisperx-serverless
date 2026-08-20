# whisperx-serverless

Worker de RunPod Serverless que transcribe y **sincroniza la letra** de una canción con
**WhisperX** (GPU). Lo consume Fototeca para generar karaoke automático (sin archivos LRC).

## Contrato
```
input:  { "audio_url": "<url R2>", "language": "es" }   # language opcional (autodetecta)
output: { "language": "es", "segments": [ { "start", "end", "text", "words":[…] } ] }
```
La salida es JSON chico → inline (no usa R2 de salida).

## Deploy (igual que demucs-serverless)
1. Push este repo a GitHub.
2. RunPod → Serverless → New Endpoint → **Deploy from a GitHub repository** → este repo.
3. GPU: **≥16GB** (large-v3 en float16 usa ~10GB). Container disk ~20GB.
4. Min workers 0, idle 5s, execution timeout 600s.
5. Copia el **Endpoint ID** → Fototeca `.env`: `WHISPERX_ENDPOINT_ID` (usa el mismo `RUNPOD_API_KEY`).

Nota: si 16GB escasea, baja a `WHISPER_MODEL=medium` (menos VRAM, casi igual de bueno para letra).
