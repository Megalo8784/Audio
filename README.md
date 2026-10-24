# transcripteur

Outil en ligne de commande, 100 % local, qui :

1. accepte n'importe quel fichier audio ou vidéo lisible par ffmpeg (wav, mp3, m4a, ogg, flac, mp4…) ;
2. produit une copie **compressée mono** (mp3 64 kbit/s par défaut, ou opus 24 kbit/s) pour l'archivage ;
3. transcrit en français avec **faster-whisper** (horodatage au mot) ;
4. identifie les locuteurs avec **pyannote.audio** (de 2 à N, détectés automatiquement) ;
5. écrit un **JSON** (détaillé, mot à mot) et un **TXT** (une ligne `[début - fin] texte` par
   segment) à côté du fichier source ;
6. si un enregistrement ne contient **aucune parole**, déplace l'audio, sa copie compressée, le JSON
   et le TXT dans un sous-dossier `a_supprimer/` (désactivable avec `--garder-silencieux`).

La transcription se fait à partir de l'audio d'origine, converti en WAV mono 16 kHz temporaire, et
**pas** à partir du MP3. La copie compressée sert uniquement à manipuler des fichiers plus légers.

## Installation (PC avec GPU NVIDIA)

**Windows 11 : suivez le guide pas à pas [docs/INSTALL_WINDOWS.md](docs/INSTALL_WINDOWS.md).**

Prérequis : Python 3.10 à 3.13, [ffmpeg](https://ffmpeg.org/download.html) dans le PATH, pilote NVIDIA récent.

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    Linux : source .venv/bin/activate

# 1. PyTorch avec CUDA 12.8+ (obligatoire pour une RTX 50xx / Blackwell)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2. L'outil + DLL CUDA pour Whisper + la diarisation
pip install -e ".[cuda,diarisation]"
```

Pour la diarisation (gratuite, mais le modèle est « gated ») :

1. créer un compte sur https://huggingface.co ;
2. accepter les conditions de https://hf.co/pyannote/speaker-diarization-community-1 ;
3. créer un jeton de lecture (https://hf.co/settings/tokens) et le mettre dans la variable `HF_TOKEN`.

Les modèles sont téléchargés au premier lancement (environ 3 Go pour `large-v3`), puis tout tourne hors ligne.

## Utilisation

```bash
transcrire appel.wav                       # -> appel.mp3 + appel.json à côté
transcrire dossier_appels/ -o sorties/     # tout un dossier
transcrire workshop.wav --min-locuteurs 3 --max-locuteurs 8
transcrire appel.wav --nb-locuteurs 2      # si le nombre est connu : plus fiable
transcrire appel.wav --prompt "Acme, Kubernetes, Dupont"   # vocabulaire à favoriser
transcrire appel.wav --compresser opus     # fichiers environ 2,5 fois plus petits qu'en mp3 64k
transcrire appel.wav --sans-locuteurs      # transcription seule, plus rapide
transcrire --help
```

Les fichiers déjà transcrits (JSON présent) sont ignorés. `--ecraser` les refait.
Avec un dossier, seuls les fichiers directement dedans sont traités (pas les sous-dossiers), et une
copie compressée `appel.mp3` à côté de `appel.wav` n'est pas retranscrite.

« Aucune parole » signifie que Whisper n'a produit aucun texte (les silences sont filtrés en amont).
Les fichiers sont **déplacés, jamais supprimés** : vérifiez `a_supprimer/` avant de le vider.

## Format JSON

```json
{
  "metadata": { "source": "appel.wav", "duration_s": 1834.2, "language": "fr", "whisper_model": "large-v3", "...": "..." },
  "speakers": ["Locuteur 1", "Locuteur 2"],
  "segments": [
    {
      "id": 0, "start": 0.52, "end": 3.1, "speaker": "Locuteur 1",
      "text": "Bonjour, vous m'entendez ?",
      "words": [ { "start": 0.52, "end": 0.9, "word": "Bonjour,", "probability": 0.98 } ]
    }
  ]
}
```

## Points d'attention

- **RTX 5080 (architecture Blackwell)** : il faut un PyTorch compilé pour CUDA 12.8 ou plus récent
  (index `cu128` ci-dessus). Le support de Blackwell par CTranslate2, le moteur de faster-whisper,
  n'a **pas été vérifié** sur ce matériel. En cas d'erreur CUDA ou cuDNN au chargement de Whisper,
  essayez `--compute-type int8_float16`, puis en dernier recours `--device cpu --compute-type int8`,
  et signalez l'erreur.
- **Audio de qualité moyenne** : `--prompt` avec les noms propres aide beaucoup. `large-v3` est plus
  précis que `large-v3-turbo`, qui est plus rapide.
- **Fichiers de 5 h** : l'audio est chargé en mémoire (environ 1,2 Go en float32 pour 5 h), ce qui
  passe avec 32 Go de RAM. La diarisation d'un fichier aussi long n'a pas encore été mesurée.
- Les étiquettes sont « Locuteur 1, 2… » dans l'ordre d'apparition. Elles ne sont **pas** reconnues
  d'un fichier à l'autre.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
