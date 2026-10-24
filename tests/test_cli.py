from types import SimpleNamespace

import pytest

from transcripteur import cli


def test_collect_folder_skips_previous_compressed_copies(tmp_path):
    for name in ["a.wav", "a.mp3", "b.mp3", "c.m4a", "c.opus", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    (tmp_path / "sous_dossier").mkdir()
    names = [f.name for f in cli.collect([tmp_path])]
    assert names == ["a.wav", "b.mp3", "c.m4a"]


def test_move_to_silent_dedups_and_avoids_overwrite(tmp_path):
    src = tmp_path / "x.wav"
    src.write_bytes(b"1")
    js = tmp_path / "x.json"
    js.write_text("{}")
    dest = tmp_path / cli.SILENT_DIR
    dest.mkdir()
    (dest / "x.wav").write_bytes(b"ancien")
    cli.move_to_silent([src, src, js], tmp_path)
    assert not src.exists() and not js.exists()
    assert sorted(p.name for p in dest.iterdir()) == ["x (1).wav", "x.json", "x.wav"]
    assert (dest / "x.wav").read_bytes() == b"ancien"


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """process() sans ffmpeg ni modèles : la transcription renvoie ce que le test fixe."""
    import numpy as np

    from transcripteur import audio, transcribe

    state = {"segments": [], "diarized": False}
    monkeypatch.setattr(audio, "to_wav16k", lambda src, dst: dst)
    monkeypatch.setattr(audio, "load_wav", lambda p: np.zeros(16000, dtype="float32"))
    monkeypatch.setattr(transcribe, "transcribe",
                        lambda *a, **k: (state["segments"], {"language": "fr",
                                                             "language_probability": 1.0}))
    src = tmp_path / "appel.wav"
    src.write_bytes(b"")
    args = SimpleNamespace(compresser="aucun", modele="m", langue="fr", device="cpu",
                           compute_type="int8", batch_size=1, prompt=None, sans_locuteurs=True,
                           modele_diarisation="d", garder_silencieux=False)
    return state, src, args


def test_process_moves_everything_when_no_speech(fake_pipeline, tmp_path):
    state, src, args = fake_pipeline
    out = cli.process(src, tmp_path, args)
    dest = tmp_path / cli.SILENT_DIR
    assert out == dest / "appel.json"
    assert sorted(p.name for p in dest.iterdir()) == ["appel.json", "appel.txt", "appel.wav"]
    assert not src.exists()


def test_process_keeps_files_with_speech(fake_pipeline, tmp_path):
    from transcripteur.transcribe import Segment

    state, src, args = fake_pipeline
    state["segments"] = [Segment(0.0, 1.0, "Bonjour", [])]
    assert cli.process(src, tmp_path, args) == tmp_path / "appel.json"
    assert src.exists()
    assert (tmp_path / "appel.txt").read_text(encoding="utf-8") == "[00:00:00 - 00:00:01] Bonjour\n"
    assert not (tmp_path / cli.SILENT_DIR).exists()
