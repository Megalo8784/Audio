"""Conversion audio via ffmpeg."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000

# Format compressé -> (extension, arguments ffmpeg). Réglages pensés pour de la voix mono.
COMPRESSION = {
    "opus": (".opus", ["-c:a", "libopus", "-b:a", "24k", "-application", "voip"]),
    "mp3": (".mp3", ["-c:a", "libmp3lame", "-b:a", "64k"]),
}
COMPRESSED_SUFFIXES = {ext for ext, _ in COMPRESSION.values()}


class AudioError(RuntimeError):
    pass


def _ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise AudioError("ffmpeg introuvable : installez-le et ajoutez-le au PATH.")
    return exe


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg a échoué ({proc.returncode}) :\n{proc.stderr[-2000:]}")


def duration(path: Path) -> float | None:
    """Durée en secondes via ffprobe, ou None si indisponible."""
    exe = shutil.which("ffprobe")
    if exe is None:
        return None
    proc = subprocess.run(
        [exe, "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True,
        text=True,
    )
    try:
        return float(json.loads(proc.stdout)["format"]["duration"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


def to_wav16k(src: Path, dst: Path) -> Path:
    """Convertit n'importe quel fichier lisible par ffmpeg en WAV PCM mono 16 kHz."""
    _run([_ffmpeg(), "-y", "-nostdin", "-i", str(src), "-vn", "-ac", "1",
          "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)])
    return dst


def compress(src: Path, out_dir: Path, fmt: str) -> Path:
    """Produit une copie compressée mono du fichier source (pour l'archivage)."""
    ext, codec_args = COMPRESSION[fmt]
    dst = out_dir / (src.stem + ext)
    if dst.resolve() == src.resolve():
        return src  # la source est déjà dans ce format : on ne l'écrase pas
    _run([_ffmpeg(), "-y", "-nostdin", "-i", str(src), "-vn", "-ac", "1", *codec_args, str(dst)])
    return dst


def load_wav(path: Path) -> np.ndarray:
    """Charge un WAV en float32 mono."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != SAMPLE_RATE:
        raise AudioError(f"Fréquence inattendue {sr} Hz (attendu {SAMPLE_RATE}).")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data
