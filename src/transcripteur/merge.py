"""Fusion transcription + diarisation : un locuteur par mot, puis découpage en interventions."""

from __future__ import annotations

import bisect

from .diarize import Turn
from .transcribe import Segment, Word


def assign_speakers(words: list[Word], turns: list[Turn]) -> None:
    """Attribue à chaque mot le locuteur qui le recouvre le plus (ou le plus proche)."""
    if not turns:
        return
    turns = sorted(turns)
    starts = [t[0] for t in turns]
    for w in words:
        # Seuls les tours commençant avant la fin du mot peuvent le recouvrir.
        hi = bisect.bisect_right(starts, w.end)
        best, best_overlap = None, 0.0
        for start, end, spk in turns[max(0, hi - 8):hi]:
            overlap = min(end, w.end) - max(start, w.start)
            if overlap > best_overlap:
                best, best_overlap = spk, overlap
        if best is None:
            best = _nearest(turns, hi, (w.start + w.end) / 2)
        w.speaker = best


def _nearest(turns: list[Turn], hi: int, t: float) -> str:
    candidates = turns[max(0, hi - 8):hi + 1]
    def dist(turn: Turn) -> float:
        start, end, _ = turn
        return 0.0 if start <= t <= end else min(abs(t - start), abs(t - end))
    return min(candidates, key=dist)[2]


def split_by_speaker(segments: list[Segment]) -> list[Segment]:
    """Découpe chaque segment Whisper là où le locuteur change."""
    out: list[Segment] = []
    for seg in segments:
        if not seg.words:
            out.append(seg)
            continue
        group: list[Word] = []
        for w in seg.words:
            if group and w.speaker != group[-1].speaker:
                out.append(_from_words(group))
                group = []
            group.append(w)
        out.append(_from_words(group))
    return out


def _from_words(words: list[Word]) -> Segment:
    text = "".join(w.text for w in words).strip()
    return Segment(words[0].start, words[-1].end, text, list(words), words[0].speaker)


def rename_speakers(segments: list[Segment]) -> list[str]:
    """Renomme SPEAKER_xx en « Locuteur N » dans l'ordre d'apparition."""
    mapping: dict[str, str] = {}
    for seg in segments:
        for item in [seg, *seg.words]:
            if item.speaker is not None:
                item.speaker = mapping.setdefault(item.speaker, f"Locuteur {len(mapping) + 1}")
    return list(mapping.values())
