import base64
import os
import re
import subprocess
import tempfile
from pathlib import Path

import runpod
import torch
import torchaudio
from chatterbox.mtl_tts import ChatterboxMultilingualTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)


def split_text(text: str, limit: int = 320):
    paragraphs = [item.strip() for item in re.split(r"\n+", text) if item.strip()]
    chunks = []
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


def decode_reference(value: str, destination: Path):
    cleaned = re.sub(r"^data:audio/[^;]+;base64,", "", value or "")
    destination.write_bytes(base64.b64decode(cleaned, validate=True))


def generate(job):
    data = job.get("input") or {}
    text = str(data.get("text") or "").strip()
    reference = str(data.get("reference_audio_base64") or "")
    speed = min(max(float(data.get("speed", 1.0)), 0.75), 1.30)
    expression = min(max(float(data.get("expression", 45)), 0), 100)
    exaggeration = 0.28 + (expression / 100.0) * 0.62

    if len(text) < 3 or not reference:
        raise ValueError("Texto e voz de referência são obrigatórios.")

    with tempfile.TemporaryDirectory(prefix="ctec_voice_") as temporary:
        root = Path(temporary)
        reference_path = root / "reference_audio"
        wav_path = root / "generated.wav"
        mp3_path = root / "generated.mp3"
        decode_reference(reference, reference_path)

        segments = []
        silence = torch.zeros(1, int(MODEL.sr * 0.28), dtype=torch.float32)
        for chunk in split_text(text):
            audio = MODEL.generate(
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
        torchaudio.save(str(wav_path), combined, MODEL.sr)
        tempo_filter = f"atempo={speed:.2f}"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
                "-filter:a", tempo_filter, "-codec:a", "libmp3lame",
                "-b:a", "192k", str(mp3_path),
            ],
            check=True,
        )
        encoded = base64.b64encode(mp3_path.read_bytes()).decode("ascii")
        return {
            "audio_base64": encoded,
            "mime_type": "audio/mpeg",
            "file_name": "ctec-voz-neural.mp3",
            "sample_rate": MODEL.sr,
        }


runpod.serverless.start({"handler": generate})
