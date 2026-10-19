"""Point d'entrée en ligne de commande : `transcrire fichier1.wav dossier/ ...`."""

from __future__ import annotations

import argparse
import gc
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__, audio
from .diarize import DEFAULT_MODEL as DEFAULT_DIARIZATION_MODEL
from .merge import assign_speakers, rename_speakers, split_by_speaker
from .output import to_dict, write_json

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
                    ".webm", ".mp4", ".mkv", ".mov", ".amr", ".3gp"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcrire",
        description="Transcrit des fichiers audio en JSON (local, GPU), avec identification des locuteurs.",
    )
    p.add_argument("entrees", nargs="+", type=Path, help="fichiers audio ou dossiers")
    p.add_argument("-o", "--sortie", type=Path, default=None,
                   help="dossier de sortie (défaut : à côté de chaque fichier source)")
    p.add_argument("--compresser", choices=["mp3", "opus", "aucun"], default="mp3",
                   help="copie compressée mono de la source (défaut : mp3)")
    p.add_argument("--ecraser", action="store_true", help="refait les fichiers déjà transcrits")

    w = p.add_argument_group("Whisper")
    w.add_argument("--modele", default="large-v3",
                   help="modèle faster-whisper (large-v3, large-v3-turbo, medium…)")
    w.add_argument("--langue", default="fr", help="code langue, ou 'auto' pour détecter")
    w.add_argument("--device", default="cuda", help="cuda ou cpu")
    w.add_argument("--compute-type", default="float16",
                   help="float16 (GPU), int8_float16, int8 (CPU)…")
    w.add_argument("--batch-size", type=int, default=16,
                   help="taille de lot ; 1 = mode séquentiel (plus lent)")
    w.add_argument("--prompt", default=None,
                   help="vocabulaire à favoriser (noms propres, jargon)")

    d = p.add_argument_group("Locuteurs")
    d.add_argument("--sans-locuteurs", action="store_true", help="désactive la diarisation")
    d.add_argument("--nb-locuteurs", type=int, default=None, help="nombre exact, s'il est connu")
    d.add_argument("--min-locuteurs", type=int, default=None)
    d.add_argument("--max-locuteurs", type=int, default=None)
    d.add_argument("--modele-diarisation", default=DEFAULT_DIARIZATION_MODEL)
    d.add_argument("--hf-token", default=None, help="jeton Hugging Face (sinon variable HF_TOKEN)")
    return p


def collect(inputs: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in inputs:
        if path.is_dir():
            files += sorted(f for f in path.iterdir()
                            if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS)
        elif path.is_file():
            files.append(path)
        else:
            print(f"Ignoré (introuvable) : {path}", file=sys.stderr)
    return files


def process(src: Path, out_dir: Path, args: argparse.Namespace) -> Path:
    from .transcribe import transcribe

    json_path = out_dir / (src.stem + ".json")
    t0 = time.monotonic()

    if args.compresser != "aucun":
        compressed = audio.compress(src, out_dir, args.compresser)
        print(f"Copie compressée : {compressed}", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="transcrire-") as tmp:
        wav = audio.to_wav16k(src, Path(tmp) / "audio.wav")
        samples = audio.load_wav(wav)
    duration = len(samples) / audio.SAMPLE_RATE
    print(f"Durée : {duration / 60:.1f} min", file=sys.stderr)

    segments, info = transcribe(
        samples,
        model_name=args.modele,
        language=None if args.langue == "auto" else args.langue,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size,
        initial_prompt=args.prompt,
    )
    gc.collect()  # libère le modèle Whisper avant de charger pyannote

    speakers: list[str] = []
    if not args.sans_locuteurs:
        from .diarize import diarize

        print("Diarisation…", file=sys.stderr)
        turns = diarize(
            samples,
            model_name=args.modele_diarisation,
            token=args.hf_token,
            device=args.device,
            num_speakers=args.nb_locuteurs,
            min_speakers=args.min_locuteurs,
            max_speakers=args.max_locuteurs,
        )
        words = [w for s in segments for w in s.words]
        assign_speakers(words, turns)
        segments = split_by_speaker(segments)
        speakers = rename_speakers(segments)

    metadata = {
        "source": str(src),
        "duration_s": round(duration, 2),
        "language": info["language"],
        "language_probability": info["language_probability"],
        "whisper_model": args.modele,
        "diarization_model": None if args.sans_locuteurs else args.modele_diarisation,
        "processing_time_s": round(time.monotonic() - t0, 1),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tool_version": __version__,
    }
    write_json(to_dict(segments, speakers, metadata), json_path)
    return json_path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    files = collect(args.entrees)
    if not files:
        print("Aucun fichier audio à traiter.", file=sys.stderr)
        return 1

    failures = 0
    for i, src in enumerate(files, 1):
        out_dir = args.sortie or src.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / (src.stem + ".json")
        print(f"\n[{i}/{len(files)}] {src}", file=sys.stderr)
        if json_path.exists() and not args.ecraser:
            print(f"Déjà transcrit, ignoré (--ecraser pour refaire) : {json_path}", file=sys.stderr)
            continue
        try:
            print(f"OK -> {process(src, out_dir, args)}", file=sys.stderr)
        except Exception as exc:  # un fichier en échec ne doit pas bloquer les suivants
            failures += 1
            print(f"ÉCHEC : {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
