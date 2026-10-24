from transcripteur.merge import assign_speakers, rename_speakers, split_by_speaker
from transcripteur.output import to_dict
from transcripteur.transcribe import Segment, Word


def words(*specs):
    return [Word(s, e, t, 0.9) for s, e, t in specs]


def test_assign_by_max_overlap():
    ws = words((0.0, 0.5, " Bonjour"), (0.9, 1.4, " oui"), (1.5, 2.0, " merci"))
    turns = [(0.0, 1.0, "SPEAKER_00"), (1.0, 3.0, "SPEAKER_01")]
    assign_speakers(ws, turns)
    # " oui" chevauche 0,1 s le premier tour et 0,4 s le second
    assert [w.speaker for w in ws] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_01"]


def test_word_outside_any_turn_gets_nearest_speaker():
    ws = words((5.0, 5.2, " euh"))
    assign_speakers(ws, [(0.0, 1.0, "A"), (6.0, 7.0, "B")])
    assert ws[0].speaker == "B"


def test_split_and_rename():
    ws = words((0.0, 0.5, " Salut"), (0.6, 1.0, " ça"), (1.1, 1.5, " va ?"), (1.6, 2.0, " Oui."))
    for w, spk in zip(ws, ["S1", "S1", "S1", "S0"]):
        w.speaker = spk
    segs = split_by_speaker([Segment(0.0, 2.0, "Salut ça va ? Oui.", ws)])
    assert [s.text for s in segs] == ["Salut ça va ?", "Oui."]
    assert rename_speakers(segs) == ["Locuteur 1", "Locuteur 2"]
    assert [s.speaker for s in segs] == ["Locuteur 1", "Locuteur 2"]
    assert segs[1].words[0].speaker == "Locuteur 2"


def test_json_shape():
    seg = Segment(0.0, 1.0, "Bonjour", words((0.0, 1.0, " Bonjour")), "Locuteur 1")
    data = to_dict([seg], ["Locuteur 1"], {"source": "x.wav"})
    assert data["segments"][0]["words"][0]["word"] == "Bonjour"
    assert data["segments"][0]["speaker"] == "Locuteur 1"


def test_nvidia_dll_dirs(tmp_path):
    from transcripteur.transcribe import nvidia_dll_dirs

    (tmp_path / "nvidia" / "cudnn" / "bin").mkdir(parents=True)
    (tmp_path / "nvidia" / "cublas" / "bin").mkdir(parents=True)
    (tmp_path / "nvidia" / "cuda_nvrtc").mkdir(parents=True)  # sans bin/ : ignoré
    found = nvidia_dll_dirs([str(tmp_path), str(tmp_path / "absent")])
    assert [d.parent.name for d in found] == ["cublas", "cudnn"]


def test_cli_fails_fast_without_hf_token(monkeypatch, tmp_path, capsys):
    from transcripteur import cli

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(cli, "process", lambda *a: (_ for _ in ()).throw(AssertionError("lancé")))
    (tmp_path / "a.wav").write_bytes(b"")
    assert cli.main([str(tmp_path / "a.wav")]) == 1
    assert "HF_TOKEN" in capsys.readouterr().err


def test_txt_has_only_timestamps_and_text():
    from transcripteur.output import to_txt

    segs = [Segment(3.4, 7.9, "Bonjour.", [], "Locuteur 1"),
            Segment(8.0, 9.0, "", []),
            Segment(3725.0, 3727.5, "Au revoir.", [], "Locuteur 2")]
    assert to_txt(segs) == "[00:00:03 - 00:00:07] Bonjour.\n[01:02:05 - 01:02:07] Au revoir.\n"
