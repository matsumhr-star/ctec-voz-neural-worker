import base64
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import runpod
import torch
import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Carregamento preguiçoso: o worker inicia primeiro e o modelo é carregado
# somente quando o ADM enviar a primeira solicitação de geração.
_MODEL = None
_MODEL_LOCK = threading.Lock()


def get_model() -> ChatterboxMultilingualTTS:
    global _MODEL

    if _MODEL is None:
        with _MODEL_LOCK:
            if _MODEL is None:
                print(f"[CTEC] Carregando ChatterboxMultilingualTTS em {DEVICE}...")
                _MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
                print("[CTEC] Modelo carregado com sucesso.")

    return _MODEL


def split_text(text: str, limit: int = 320) -> list[str]:
    paragraphs = [item.strip() for item in re.split(r"\n+", text) if item.strip()]
    chunks: list[str] = []

    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?;:])\s+", paragraph)
        current = ""

        for sentence in sentences:
            candidate = f"{current} {sentence}".strip()

            if current and len(candidate) > limit:
                chunks.append(current)
                current = sentence
            else:
                current = candidate

        if current:
            chunks.append(current)

    return chunks


def decode_reference(value: str, destination: Path) -> None:
    cleaned = re.sub(r"^data:audio/[^;]+;base64,", "", value or "")
    destination.write_bytes(base64.b64decode(cleaned, validate=True))


def generate(job: dict[str, Any]) -> dict[str, Any]:
    data = job.get("input") or {}

    text = str(data.get("text") or "").strip()
    reference = str(data.get("reference_audio_base64") or "").strip()
    speed = min(max(float(data.get("speed", 1.0)), 0.75), 1.30)
    expression = min(max(float(data.get("expression", 45)), 0), 100)
    exaggeration = 0.28 + (expression / 100.0) * 0.62

    if len(text) < 3:
        raise ValueError("O texto precisa ter pelo menos 3 caracteres.")

    if not reference:
        raise ValueError("A voz de referência é obrigatória.")

    model = get_model()
    chunks = split_text(text)

    if not chunks:
        raise ValueError("Não foi possível dividir o texto para geração.")

    with tempfile.TemporaryDirectory(prefix="ctec_voice_") as temporary:
        root = Path(temporary)
        reference_path = root / "reference_audio.wav"
        wav_path = root / "generated.wav"
        mp3_path = root / "generated.mp3"

        decode_reference(reference, reference_path)

        segments: list[torch.Tensor] = []
        silence = torch.zeros(1, int(model.sr * 0.28), dtype=torch.float32)

        for chunk in chunks:
            audio = model.generate(
                chunk,
                language_id="pt",
                audio_prompt_path=str(reference_path),
                exaggeration=exaggeration,
                cfg_weight=0.5,
            ).detach().cpu()

            if audio.ndim == 1:
                audio = audio.unsqueeze(0)

            segments.extend([audio, silence])

        combined = torch.cat(segments, dim=1)
        torchaudio.save(str(wav_path), combined, model.sr)

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-i",
                str(wav_path),
                "-filter:a",
                f"atempo={speed:.2f}",
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(mp3_path),
            ],
            check=True,
        )

        encoded = base64.b64encode(mp3_path.read_bytes()).decode("ascii")

        return {
            "audio_base64": encoded,
            "mime_type": "audio/mpeg",
            "file_name": "ctec-voz-neural.mp3",
            "sample_rate": model.sr,
        }

runpod.serverless.start({"handler": generate})
