"""RunPod Serverless — Transcripción + alineación de letra con WhisperX (GPU).

Contrato:
  input:  { "audio_url": "<url R2>", "language": "es" (opcional) }
  output: { "language": "es",
            "segments": [ { "start": 1.2, "end": 4.8, "text": "línea…",
                            "words": [ { "start":1.2, "end":1.4, "word":"línea" }, … ] }, … ] }

La salida es texto (JSON chico) → se devuelve inline (no necesita R2 de salida).
Lo ideal es transcribir el STEM DE VOZ (más limpio), pero también sirve la mezcla.
"""
import os
import tempfile
import urllib.request

import runpod
import whisperx

DEVICE = "cuda"
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")
BATCH = int(os.environ.get("WHISPER_BATCH", "16"))

_model = None
_align = {}  # language_code -> (model, metadata)


def _get_model():
    global _model
    if _model is None:
        _model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE)
    return _model


def handler(job):
    inp = job.get("input") or {}
    audio_url = inp.get("audio_url")
    lang = inp.get("language")  # None = autodetecta
    if not audio_url:
        return {"error": "falta 'audio_url'"}
    try:
        d = tempfile.mkdtemp()
        path = os.path.join(d, "audio")
        urllib.request.urlretrieve(audio_url, path)

        audio = whisperx.load_audio(path)
        result = _get_model().transcribe(audio, batch_size=BATCH, language=lang)
        language = result.get("language", lang or "es")

        # Alineación palabra por palabra (cachea el modelo por idioma).
        try:
            if language not in _align:
                _align[language] = whisperx.load_align_model(language_code=language, device=DEVICE)
            amodel, meta = _align[language]
            aligned = whisperx.align(result["segments"], amodel, meta, audio, DEVICE,
                                     return_char_alignments=False)
            segs_src = aligned["segments"]
        except Exception:
            # Si no hay modelo de alineación para el idioma, usa los segmentos crudos.
            segs_src = result["segments"]

        segments = []
        for s in segs_src:
            words = [
                {"start": w.get("start"), "end": w.get("end"), "word": w.get("word", "").strip()}
                for w in s.get("words", []) if w.get("word")
            ]
            segments.append({
                "start": s.get("start"),
                "end": s.get("end"),
                "text": (s.get("text") or "").strip(),
                "words": words,
            })
        return {"language": language, "segments": segments}
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}


runpod.serverless.start({"handler": handler})
