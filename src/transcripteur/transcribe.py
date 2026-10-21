"""Transcription avec faster-whisper (horodatage au mot)."""

from __future__ import annotations

import os
import site
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float
    speaker: str | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    speaker: str | None = None


def transcribe(
    audio: np.ndarray,
    *,
    model_name: str = "large-v3",
    language: str | None = "fr",
    device: str = "cuda",
    compute_type: str = "float16",
    batch_size: int = 16,
    beam_size: int = 5,
    initial_prompt: str | None = None,
) -> tuple[list[Segment], dict]:
    """Retourne les segments Whisper et des infos (langue détectée, etc.)."""
    if sys.platform == "win32" and device == "cuda":
        register_cuda_dlls()
    from faster_whisper import BatchedInferencePipeline, WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    options = dict(
        language=language,
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,  # coupe les silences : limite les hallucinations sur l'audio bruité
        initial_prompt=initial_prompt,
    )
    if batch_size > 1:
        segments_iter, info = BatchedInferencePipeline(model=model).transcribe(
            audio, batch_size=batch_size, **options
        )
    else:
        # Sans contexte précédent : évite les boucles de répétition sur les longs fichiers.
        segments_iter, info = model.transcribe(
            audio, condition_on_previous_text=False, **options
        )

    total = len(audio) / SAMPLE_RATE
    segments: list[Segment] = []
    for seg in segments_iter:
        words = [Word(w.start, w.end, w.word, w.probability) for w in (seg.words or [])]
        segments.append(Segment(seg.start, seg.end, seg.text.strip(), words))
        _progress("Transcription", seg.end, total)
    _progress("Transcription", total, total, done=True)

    return segments, {
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
    }


def nvidia_dll_dirs(site_dirs: list[str]) -> list[Path]:
    """Dossiers bin/ des paquets pip nvidia-* (cuBLAS, cuDNN) présents dans site-packages."""
    return [d for s in site_dirs for d in sorted(Path(s).glob("nvidia/*/bin")) if d.is_dir()]


def register_cuda_dlls() -> None:
    """Windows : rend les DLL cuBLAS/cuDNN installées par pip visibles pour CTranslate2.

    CTranslate2 n'embarque pas ces DLL sous Windows ; cuDNN charge aussi ses
    sous-bibliothèques lui-même, d'où l'ajout au PATH en plus de add_dll_directory.
    """
    dirs = nvidia_dll_dirs(site.getsitepackages() + [site.getusersitepackages()])
    for d in dirs:
        os.add_dll_directory(str(d))
    os.environ["PATH"] = os.pathsep.join([*map(str, dirs), os.environ.get("PATH", "")])


def _progress(label: str, current: float, total: float, done: bool = False) -> None:
    pct = 100.0 if done or total <= 0 else min(100.0, 100.0 * current / total)
    sys.stderr.write(f"\r{label} : {pct:5.1f} %")
    if done:
        sys.stderr.write("\n")
    sys.stderr.flush()
