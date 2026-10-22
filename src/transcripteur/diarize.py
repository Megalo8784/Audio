"""Diarisation (qui parle quand) avec pyannote.audio 4.x."""

from __future__ import annotations

import os

import numpy as np

from .audio import SAMPLE_RATE

DEFAULT_MODEL = "pyannote/speaker-diarization-community-1"

Turn = tuple[float, float, str]  # (début, fin, locuteur)


class DiarizationError(RuntimeError):
    pass


def diarize(
    audio: np.ndarray,
    *,
    model_name: str = DEFAULT_MODEL,
    token: str | None = None,
    device: str = "cuda",
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[Turn]:
    """Retourne les tours de parole sans chevauchement, triés par début."""
    # pyannote 4 envoie par défaut des statistiques d'usage (durée des fichiers, nombre de
    # locuteurs demandé) à otel.pyannote.ai : désactivé sauf choix explicite de l'utilisateur.
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise DiarizationError(
            "pyannote.audio n'est pas installé : pip install -e .[diarisation]"
        ) from exc

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise DiarizationError(
            "Jeton Hugging Face manquant (variable HF_TOKEN ou --hf-token). "
            f"Acceptez aussi les conditions du modèle sur https://hf.co/{model_name}"
        )

    pipeline = Pipeline.from_pretrained(model_name, token=token)
    if pipeline is None:
        raise DiarizationError(
            f"Impossible de charger {model_name} : conditions d'utilisation acceptées sur Hugging Face ?"
        )
    pipeline.to(torch.device(device))

    # Audio passé en mémoire : évite un second décodage du fichier par pyannote.
    waveform = torch.from_numpy(np.ascontiguousarray(audio)).unsqueeze(0)
    output = pipeline(
        {"waveform": waveform, "sample_rate": SAMPLE_RATE},
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    # pyannote 4 : la version "exclusive" (sans chevauchement) est prévue pour la transcription.
    annotation = getattr(output, "exclusive_speaker_diarization", output)
    turns = [(t.start, t.end, spk) for t, _, spk in annotation.itertracks(yield_label=True)]
    return sorted(turns)
