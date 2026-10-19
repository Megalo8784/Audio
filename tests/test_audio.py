import shutil
import subprocess

import pytest

from transcripteur import audio

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg absent")


@pytest.fixture
def stereo_44k(tmp_path):
    src = tmp_path / "appel.wav"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=frequency=440:duration=2:sample_rate=44100", "-ac", "2", str(src)],
                   check=True)
    return src


def test_to_wav16k_is_mono_16k(stereo_44k, tmp_path):
    out = audio.to_wav16k(stereo_44k, tmp_path / "out.wav")
    samples = audio.load_wav(out)
    assert samples.ndim == 1
    assert abs(len(samples) / audio.SAMPLE_RATE - 2.0) < 0.05


@pytest.mark.parametrize("fmt", ["mp3", "opus"])
def test_compress_is_smaller(stereo_44k, tmp_path, fmt):
    out_dir = tmp_path / "sortie"
    out_dir.mkdir()
    out = audio.compress(stereo_44k, out_dir, fmt)
    assert out.suffix == "." + fmt
    assert out.stat().st_size < stereo_44k.stat().st_size


def test_compress_never_overwrites_source(tmp_path, stereo_44k):
    mp3 = audio.compress(stereo_44k, tmp_path, "mp3")
    size = mp3.stat().st_size
    assert audio.compress(mp3, tmp_path, "mp3") == mp3
    assert mp3.stat().st_size == size
