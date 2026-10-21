# Installation pas à pas — Windows 11 + GPU NVIDIA (RTX 50xx)

Durée : environ 30 à 45 minutes, surtout du téléchargement (environ 8 Go au total).
Toutes les commandes se tapent dans **PowerShell** : menu Démarrer → « Terminal ».

> Après chaque étape, faites la **vérification** indiquée avant de passer à la suite.
> Si elle échoue, arrêtez-vous et consultez le dépannage en bas de page.

---

## Étape 1 — Pilote NVIDIA

1. Mettez à jour le pilote avec l'application **NVIDIA App**, ou depuis https://www.nvidia.com/fr-fr/drivers/.
2. Vérification :
   ```powershell
   nvidia-smi
   ```
   ✅ On doit voir `NVIDIA GeForce RTX 5080` et, en haut à droite, `CUDA Version: 12.8` ou plus.

Il n'est **pas** nécessaire d'installer le « CUDA Toolkit » : tout ce qu'il faut arrive via pip.

## Étape 2 — Python 3.12, Git, FFmpeg

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id Git.Git
winget install -e --id Gyan.FFmpeg
winget install -e --id Microsoft.VCRedist.2015+.x64
```

**Fermez puis rouvrez le Terminal** pour que le PATH soit mis à jour.

Vérification :
```powershell
py -3.12 --version     # Python 3.12.x
git --version
ffmpeg -version        # première ligne : ffmpeg version ...
```

Python 3.12 est recommandé plutôt que 3.13 ou 3.14 : c'est la version la plus sûre pour toutes les
bibliothèques utilisées.

## Étape 3 — Autoriser l'activation de l'environnement Python

PowerShell bloque par défaut le script d'activation. À faire une seule fois :
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
Répondez `O` (Oui) si une confirmation est demandée.

## Étape 4 — Récupérer le code

Placez le projet dans un chemin court, pour éviter la limite de 260 caractères de Windows :
```powershell
mkdir C:\dev -Force
cd C:\dev
git clone -b claude/audio-transcription-app-iur0rj https://github.com/megalo8784/audio.git
cd audio
```
Le dépôt est privé : Git ouvre une fenêtre de connexion GitHub la première fois.

## Étape 5 — Environnement virtuel

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```
✅ L'invite commence maintenant par `(.venv)`.

## Étape 6 — PyTorch avec CUDA

```powershell
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
```
Le téléchargement fait environ 3 Go. Vérification :
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
✅ Attendu : une version suivie de `+cu128`, puis `True NVIDIA GeForce RTX 5080`.

- Si pip répond `No matching distribution` : allez sur https://pytorch.org/get-started/locally/,
  choisissez *Stable / Windows / Pip / Python / CUDA 12.8 ou plus récent*, et lancez la commande affichée.
- ❌ Si la version finit par `+cpu` ou si vous obtenez `False` : c'est la version sans GPU.
  Relancez la commande avec `--force-reinstall`.

## Étape 7 — Installer l'outil

```powershell
pip install -e ".[cuda,diarisation]"
```
Le téléchargement fait environ 1,5 Go : les DLL cuBLAS et cuDNN pour Whisper, plus pyannote.

Vérification :
1. Relancez la commande de vérification de l'étape 6. Elle doit toujours afficher `+cu128 True`.
   pip ne doit pas avoir remplacé PyTorch par la version CPU.
2. Lancez ceci :
   ```powershell
   transcrire --help
   ```

## Étape 8 — Premier test : transcription seule

Prenez un enregistrement **court** (1 à 2 minutes) :
```powershell
transcrire C:\chemin\vers\appel_court.wav --sans-locuteurs --compresser aucun
```
Au premier lancement, le modèle `large-v3` se télécharge (environ 3 Go).
✅ Un fichier `appel_court.json` apparaît à côté de l'audio.

Faites ce test **avant** la diarisation. En cas d'erreur, on saura que le problème vient de Whisper.

## Étape 9 — Accès au modèle de diarisation (gratuit)

1. Créez un compte sur https://huggingface.co.
2. Ouvrez https://huggingface.co/pyannote/speaker-diarization-community-1 et acceptez les conditions
   d'utilisation (formulaire en haut de page).
3. Créez un jeton sur https://huggingface.co/settings/tokens → *Create new token* → type **Read**.
   Copiez-le : il commence par `hf_`.
4. Enregistrez-le pour votre compte Windows :
   ```powershell
   setx HF_TOKEN "hf_VOTRE_JETON"
   ```
5. **Fermez et rouvrez le Terminal**, puis :
   ```powershell
   cd C:\dev\audio
   .\.venv\Scripts\Activate.ps1
   ```

## Étape 10 — Test complet

```powershell
transcrire C:\chemin\vers\appel_court.wav --ecraser
```
✅ Le JSON contient maintenant `"speakers": ["Locuteur 1", "Locuteur 2", ...]`, et un fichier
`appel_court.mp3` apparaît à côté de l'audio.

Avertissements **sans importance**, que vous pouvez ignorer :
- `torchcodec is not installed correctly…` : l'outil n'utilise pas torchcodec, il fournit l'audio
  déjà décodé à pyannote ;
- `huggingface_hub … symlinks` : c'est seulement un cache disque moins économe. Pour le supprimer,
  activez le *Mode développeur* de Windows.

---

## Utilisation au quotidien

```powershell
cd C:\dev\audio
.\.venv\Scripts\Activate.ps1
transcrire D:\Appels\                           # tout un dossier
transcrire workshop.wav --min-locuteurs 3 --max-locuteurs 8
```

Mise à jour du code :
```powershell
cd C:\dev\audio
git pull
pip install -e ".[cuda,diarisation]"
```

## Dépannage

| Symptôme | Piste |
|---|---|
| `winget` introuvable | Installez « App Installer » depuis le Microsoft Store. |
| `ffmpeg` introuvable après installation | Fermez et rouvrez le Terminal. Sinon, redémarrez Windows. |
| `Activate.ps1 cannot be loaded` | Refaites l'étape 3. |
| `Could not locate cudnn_ops64_9.dll` ou `cublas64_12.dll` (étape 8) | Vérifiez que l'étape 7 inclut bien `[cuda,...]`. Relancez `pip install -e ".[cuda,diarisation]"`. |
| `CUDA failed with error … no kernel image` (étape 8) | Incompatibilité entre le moteur de Whisper et la RTX 50xx. Essayez `--compute-type int8_float16`. En dernier recours : `--device cpu --compute-type int8` (lent). Envoyez-moi le message complet. |
| `401` / `403` / `gated repo` (étape 10) | Conditions non acceptées sur la page du modèle, ou jeton absent : vérifiez avec `echo $env:HF_TOKEN`. |
| `CUDA out of memory` | Réduisez `--batch-size` à 8 ou 4. |

Sur un problème non listé, copiez **tout** le message d'erreur et envoyez-le-moi.
