"""Écriture du résultat en JSON."""

from __future__ import annotations

import json
from pathlib import Path

from .transcribe import Segment


def to_dict(segments: list[Segment], speakers: list[str], metadata: dict) -> dict:
    return {
        "metadata": metadata,
        "speakers": speakers,
        "segments": [
            {
                "id": i,
                "start": round(s.start, 2),
                "end": round(s.end, 2),
                "speaker": s.speaker,
                "text": s.text,
                "words": [
                    {
                        "start": round(w.start, 2),
                        "end": round(w.end, 2),
                        "word": w.text.strip(),
                        "probability": round(w.probability, 3),
                    }
                    for w in s.words
                ],
            }
            for i, s in enumerate(segments)
        ],
    }


def write_json(data: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)
