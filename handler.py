"""RunPod Serverless — Letra: transcripción con WhisperX y traducción con MADLAD-400 (GPU).

Dos tareas en el mismo worker para no pagar dos arranques en frío:

  task "transcribe" (por defecto)
    input:  { "audio_url": "<url R2>", "language": "es" (opcional) }
    output: { "language": "es",
              "segments": [ { "start": 1.2, "end": 4.8, "text": "línea…",
                              "words": [ { "start":1.2, "end":1.4, "word":"línea" }, … ] }, … ] }

  task "translate"
    input:  { "task": "translate", "lines": ["verso 1", "verso 2", …],
              "targets": ["en","pt"], "source": "es" (informativo) }
    output: { "translations": { "en": ["verse 1", …], "pt": [ … ] } }

La salida es texto (JSON chico) → se devuelve inline (no necesita R2 de salida).
Lo ideal es transcribir el STEM DE VOZ (más limpio), pero también sirve la mezcla.

La traducción va línea por línea y DEVUELVE EXACTAMENTE LAS MISMAS LÍNEAS que entran:
el karaoke reutiliza los tiempos del original, así que si se pierde o se junta una
línea, el subtítulo queda corrido contra la música.
"""
import os
import tempfile
import urllib.request

import runpod
import whisperx

DEVICE = "cuda"
# MADLAD-400: traductor multilingüe de Google, Apache 2.0 y cualquier idioma a
# cualquier idioma con un solo modelo. Se descarta NLLB-200 a propósito: sus pesos
# son CC-BY-NC y Ravit cobra suscripción.
MADLAD_MODEL = os.environ.get("MADLAD_MODEL", "google/madlad400-3b-mt")
TRANSLATE_BATCH = int(os.environ.get("TRANSLATE_BATCH", "16"))
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")
COMPUTE = os.environ.get("WHISPER_COMPUTE", "float16")
BATCH = int(os.environ.get("WHISPER_BATCH", "16"))

_model = None
_align = {}  # language_code -> (model, metadata)
_mt = None   # (tokenizer, modelo) de MADLAD, cargado la primera vez que se traduce


def _get_model():
    global _model
    if _model is None:
        _model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE)
    return _model


def _liberar_whisper():
    """Suelta Whisper de la VRAM. Los dos modelos juntos no caben en 16 GB."""
    global _model, _align
    import gc
    import torch
    _model = None
    _align = {}
    gc.collect()
    torch.cuda.empty_cache()


def _get_mt():
    """Carga MADLAD una sola vez por worker (pesa; el cold start ya lo pagó).

    Si la GPU no da para tener Whisper y MADLAD al tiempo, suelta Whisper y
    reintenta: así el worker sirve en tarjetas de 16 GB, pagando una recarga
    cuando el mismo worker vuelve a transcribir."""
    global _mt
    if _mt is None:
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        tok = AutoTokenizer.from_pretrained(MADLAD_MODEL)

        def cargar():
            m = T5ForConditionalGeneration.from_pretrained(
                MADLAD_MODEL, torch_dtype=torch.float16, device_map=DEVICE,
            )
            m.eval()
            return m

        try:
            mdl = cargar()
        except torch.cuda.OutOfMemoryError:
            _liberar_whisper()
            mdl = cargar()
        _mt = (tok, mdl)
    return _mt


def _translate(lines, targets):
    """Traduce la MISMA lista de líneas a cada idioma destino.

    MADLAD elige el destino con un token al principio del texto (`<2en> …`), así
    que no hace falta un modelo por par de idiomas."""
    import torch

    limpias = [(l or "").strip() for l in lines]
    tok, mdl = _get_mt()
    salida = {}

    for dest in targets:
        traducidas = []
        for i in range(0, len(limpias), TRANSLATE_BATCH):
            lote = limpias[i:i + TRANSLATE_BATCH]
            # Las líneas vacías no se mandan al modelo, pero conservan su lugar.
            idx = [j for j, t in enumerate(lote) if t]
            res = [""] * len(lote)
            if idx:
                prompts = [f"<2{dest}> {lote[j]}" for j in idx]
                enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                          max_length=256).to(DEVICE)
                with torch.inference_mode():
                    gen = mdl.generate(**enc, max_new_tokens=128, num_beams=1)
                for k, j in enumerate(idx):
                    res[j] = tok.decode(gen[k], skip_special_tokens=True).strip()
            traducidas.extend(res)
        salida[dest] = traducidas

    return salida


def handler(job):
    inp = job.get("input") or {}

    if (inp.get("task") or "transcribe") == "translate":
        lines = inp.get("lines") or []
        targets = inp.get("targets") or []
        if not lines:
            return {"error": "falta 'lines'"}
        if not targets:
            return {"error": "falta 'targets'"}
        try:
            return {"translations": _translate(lines, targets)}
        except Exception as exc:  # noqa: BLE001
            return {"error": f"{type(exc).__name__}: {exc}"}

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
