# SoilModel — how to push to GitHub and deploy on Railway

This repository contains a Flask app that loads a Keras `.h5` model and exposes a `/predict` endpoint.

Quick notes before pushing:

- The model file `final_model_20251027_131112.h5` in this repo is large (~166 MB). GitHub blocks files > 100 MB. You must either use Git LFS or store the model externally (S3/GCS) and download it at runtime.
- Railway will run `gunicorn` using the `Procfile`.

Files added:

- `requirements.txt` — minimal Python dependencies.
- `Procfile` — start command for Railway.
- `.gitignore` — ignores the model file by default.

1) Initialize repo and push to GitHub (PowerShell)

If you want to keep the model in Git and your repo is small, use Git LFS (recommended for this model size):

Install and configure Git LFS (one-time):

```powershell
# install LFS (only needed once)
git lfs install

# track .h5 files
git lfs track "*.h5"

# confirm .gitattributes created
type .gitattributes
```

Then initialize and push (replace <your-remote-url> with your GitHub repo URL):

```powershell
cd D:\Vscode\SoilModel
git init
git add .
git commit -m "Initial commit: Flask app + deployment files"
# add GitHub remote (create an empty repo on GitHub first)
git remote add origin https://github.com/<your-username>/<your-repo>.git
git branch -M main
git push -u origin main
```

If you prefer not to use Git LFS, remove the model from the repo and upload it to cloud storage (S3/GCS) and download at runtime. See below for a sample download snippet.

2) Deploy on Railway

- Create an account at https://railway.app/ (or sign in).
- Create a new project and choose "Deploy from GitHub".
- Connect your GitHub account and select the repository.
- Railway will detect the `Procfile` and `requirements.txt` and should build automatically.
- The app will be available at a Railway URL. Railway provides a PORT env var automatically; `Procfile` uses it.

Notes for model storage options

Option A — Git LFS (store in repo):
- Use `git lfs track "*.h5"` locally. Commit the `.gitattributes` file. Push. Railway will fetch LFS files during build if LFS is enabled for the runner. (If Railway build runner does not fetch LFS automatically, prefer external storage.)

Option B — External storage (recommended for reliability):
- Upload `final_model_20251027_131112.h5` to S3 or Google Cloud Storage and change your app to download the model at startup if it doesn't exist locally. Example snippet (boto3 for S3):

```python
# Example: download model from S3 at startup (add to top of app.py)
import os
import boto3

MODEL_PATH = 'final_model_20251027_131112.h5'
S3_BUCKET = os.environ.get('MODEL_S3_BUCKET')
S3_KEY = 'models/final_model_20251027_131112.h5'

if S3_BUCKET and not os.path.exists(MODEL_PATH):
    s3 = boto3.client('s3')
    s3.download_file(S3_BUCKET, S3_KEY, MODEL_PATH)

# ensure boto3 is added to requirements if using this approach
```

If you use cloud storage, set the appropriate Railway environment variables (e.g., `MODEL_S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) in the Railway project settings.

Troubleshooting and tips

- TensorFlow is large; Railway builds may be slow or hit memory limits. Consider using a smaller model or TensorFlow Lite if build fails.
- If a Railway build fails installing `tensorflow`, try using a `requirements.txt` pin such as `tensorflow==2.13.0` or `tensorflow-cpu` depending on compatibility.
- If you get errors related to missing GPU libs, prefer the CPU-only package.

If you want, I can:

- Add `git lfs` support files (.gitattributes) and show the exact commit flow.
- Modify `app.py` to optionally download the model from S3/GCS at runtime and add `boto3` to `requirements.txt`.

---
Last checked model size: ~166 MB — do you want to keep it in the repo (use Git LFS) or host externally? Reply with your choice and I will add the exact commands or change `app.py` to download the model at startup.

## Release & GitHub hosting (recommended for this repo)

If you can't attach the `.h5` directly in the GitHub repo (due to size or LFS pointer behavior), the recommended GitHub-only approach is to publish the `.h5` as a GitHub Release asset and point `MODEL_URL` to the release download URL. Releases serve the binary directly and avoid Git LFS pointer downloads that break Keras/HDF5.

Below are step-by-step options (web UI and CLI) and verification steps.

### 1) Using the GitHub web UI

- Go to: https://github.com/Cabarrubias-Ryan/SoilSnapModel/releases
- Click `Draft a new release` (or `Create a new release`).
- Fill Tag version (e.g. `v1.0`), Title (e.g. `Soil model v1.0`) and a short description.
- Drag & drop `final_model_20251027_131112.h5` into the "Attach binaries by dropping them here or selecting them" area.
- Click `Publish release`.

After the release is published the asset will be available at a URL like:

```
https://github.com/Cabarrubias-Ryan/SoilSnapModel/releases/download/v1.0/final_model_20251027_131112.h5
```

Use that URL as the `MODEL_URL` environment variable in your deployment.

### 2) Using GitHub CLI (recommended if the web UI fails)

Install and authenticate `gh` if you don't already have it (see https://cli.github.com/).

From the folder that contains `final_model_20251027_131112.h5` run (PowerShell):

```powershell
gh auth login
gh release create v1.0 final_model_20251027_131112.h5 --title "Soil model v1.0" --notes "Final model binary" --repo Cabarrubias-Ryan/SoilSnapModel
```

If the release tag already exists, upload the asset only:

```powershell
gh release upload v1.0 final_model_20251027_131112.h5 --repo Cabarrubias-Ryan/SoilSnapModel
```

To print the direct download URL for the uploaded asset:

```powershell
gh release view v1.0 --repo Cabarrubias-Ryan/SoilSnapModel --json assets -q '.assets[] | .browser_download_url'
```

### 3) Verify the asset URL serves a binary (quick check)

From PowerShell you can use curl to ensure the response is binary-like (this fetches only the first 64 bytes):

```powershell
(curl -Method Head "https://github.com/Cabarrubias-Ryan/SoilSnapModel/releases/download/v1.0/final_model_20251027_131112.h5").Headers
```

Or download a small chunk to inspect the header (HDF5 files start with the 8-byte signature `\x89HDF\r\n\x1a\n`):

```powershell
(curl "https://github.com/Cabarrubias-Ryan/SoilSnapModel/releases/download/v1.0/final_model_20251027_131112.h5" -OutFile tmp_head.bin)
[System.IO.File]::ReadAllBytes('tmp_head.bin')[0..7]  # should start with 89 48 44 46 0D 0A 1A 0A
Remove-Item tmp_head.bin
```

If you see a small text file (Git LFS pointer) instead of the HDF5 signature, do not use that URL — upload the real binary as a release asset instead.

### 4) Set `MODEL_URL` in Railway (web UI)

- Open your Railway project → Settings → Variables (Environment variables).
- Add a variable `MODEL_URL` with the value set to the release download URL shown above.

Railway will use that environment variable at runtime. On startup the app will download the `.h5` and load it.

### 5) Notes & troubleshooting

- If the web upload fails or times out, use `gh` (it uses resumable/chunked uploads and is usually more reliable).
- If you see a log message like `version https://git-lfs.github.com/spec/v1` in your app logs, that means your `MODEL_URL` pointed to a Git LFS pointer file — replace it with the release URL (or another direct-binary URL).
- Release assets accept large files (generally up to multiple GB). If your file is extremely large, consider S3/GCS instead.

### 6) Optional: add a note in the repo

Consider adding a short note in this README or a `docs/RELEASE.md` mentioning the release tag you created and the `MODEL_URL` used, so future deploys know where the binary is hosted.

---
With the release uploaded and `MODEL_URL` set, the app should download and load the model automatically at startup.

Example `MODEL_URL` for this project (already published):

```
https://github.com/Cabarrubias-Ryan/SoilSnapModel/releases/download/v1.0/final_model_20251027_131112.h5
```

You can set that value as the `MODEL_URL` environment variable in your deployment. If you prefer, the app's default `MODEL_URL` is already set to this release URL so you don't need to set the variable when running the provided code.
