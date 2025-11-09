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
