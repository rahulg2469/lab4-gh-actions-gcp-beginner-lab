# Lab 3: GitHub Actions + GCP — Breast Cancer Classifier with a Gated, Versioned Pipeline

This lab trains a **Gradient Boosting** classifier on the **Breast Cancer Wisconsin** dataset inside GitHub Actions. The model is published to Google Cloud Storage (GCS) **only if it passes a quality gate**.

It is based on [raminmohammadi/MLOps – Labs/Github_Labs/Lab3](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Github_Labs/Lab3).

---

## What the original lab does

One workflow with **one job** that runs daily or manually. That job:
- checks out the code
- installs dependencies
- authenticates to GCP with a service account JSON key
- runs a script that trains a RandomForest on Iris and uploads it to GCS with a timestamped name

There are no tests, and the model is uploaded no matter how it scores.

## What I changed

### Dataset and model
| | Original | This version |
|---|---|---|
| Dataset | Iris (150 rows, 3 classes) | Breast Cancer Wisconsin (569 rows, 30 features, malignant vs benign) |
| Model | RandomForestClassifier | GradientBoostingClassifier |
| Metrics | Accuracy | Accuracy, malignant recall, precision, F1, ROC-AUC |

### Pipeline (the core of the lab)
1. **Three chained jobs instead of one: `test` → `train` → `upload`.**
   - `test` runs a pytest suite. If any test fails, nothing trains.
   - `train` trains and evaluates the model. It has **no GCP access**.
   - `upload` is the **only job that uses the GCP key**.
   - The trained model is passed between jobs as a GitHub Actions artifact.
2. **Quality gate.** The training step fails (exit code 1) if accuracy **or** recall on the malignant class is below its threshold. Malignant recall is gated on its own because missing a cancer case is worse than a false alarm. A failed gate stops the pipeline, so the bad model never reaches GCS.
3. **Versioned uploads.** Every run uploads `model.joblib` and `metrics.json` to `gs://<bucket>/models/<commit-sha>-run<N>/`. It also overwrites `gs://<bucket>/models/latest/`, so you always know which code produced which model and how it scored.
4. **More triggers.**
   - Pushes to `main` run the pipeline automatically.
   - Pull requests run `test` and `train` but **never** `upload`.
   - Manual runs accept inputs: `n_estimators`, `learning_rate`, `max_depth`, `min_accuracy`, `min_recall`.
   - The daily cron is kept from the original.

The bucket name now comes from a GitHub repository variable (`GCS_BUCKET_NAME`) instead of being hardcoded.

## Repo structure

```
.
├── .github/workflows/train-and-upload.yml   # the pipeline
├── train_and_save_model.py                  # train, evaluate, quality gate, write artifacts
├── tests/test_train.py                      # pytest suite (6 tests)
├── requirements.txt
└── README.md
```

---

## 1. Install prerequisites

| Tool | Version | Link |
|---|---|---|
| Git | any recent | https://git-scm.com/downloads |
| Python | **3.10** (CI uses 3.10) | https://www.python.org/downloads/ |
| GitHub account | — | https://github.com |
| Google Cloud project with billing enabled | — | https://console.cloud.google.com |

Check the installs:
```bash
git --version
python --version   # 3.10.x
```

## 2. Run it locally (no GCP needed)

```bash
git clone https://github.com/<your-username>/<this-repo>.git
cd <this-repo>

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

pytest -v                        # expect: 6 passed
python train_and_save_model.py   # trains and writes ./artifacts/
```

A successful run prints the metrics (accuracy ≈ 0.956, malignant recall ≈ 0.905) and ends with:
```
✅ Quality gate passed
```
It creates `artifacts/model.joblib` and `artifacts/metrics.json`.

To try different settings:
```bash
python train_and_save_model.py --n-estimators 200 --learning-rate 0.05 --max-depth 2
```

## 3. Set up GCP (in the Cloud Console)

1. **Create a bucket.** Go to **Cloud Storage → Buckets → Create**. Pick a globally unique name and keep the defaults. Note the name.
2. **Create a service account.** Go to **IAM & Admin → Service Accounts → Create service account**. Name it e.g. `gh-actions-trainer`.
3. **Give it access.** On the bucket's **Permissions** tab, click **Grant access**. Add the service account's email with the role **Storage Object Admin**.
   - If you can't edit bucket permissions, grant **Storage Admin** at the project level instead, under **IAM & Admin → IAM**.
4. **Create a key.** Open the service account and go to **Keys → Add key → Create new key → JSON**. A `.json` file downloads.
   - Keep it private and **never commit it**. The `.gitignore` blocks common key filenames.

## 4. Configure GitHub

In your repo, go to **Settings → Secrets and variables → Actions**:

| Tab | Name | Value |
|---|---|---|
| Secrets | `GCP_SA_KEY` | Entire contents of the JSON key file |
| Variables | `GCS_BUCKET_NAME` | Your bucket name |

## 5. Run the pipeline

Any of these will start a run:
- **Push to `main`.** Any push starts a run.
- **Manual run.** Go to **Actions → Train, gate, and upload model to GCS → Run workflow**. You can change the inputs here.
- **Daily cron** at 00:00 UTC.

## 6. What success looks like

- **Actions tab:** three green jobs in a chain, `test` → `train` → `upload`.
- **`train` job log:** the metrics JSON followed by `✅ Quality gate passed`.
- **`upload` job log:** `✅ Uploaded to gs://<bucket>/models/<sha>-run<N>/ ...`
- **GCS console:** under `models/` in your bucket you'll see one folder per run plus `latest/`. Each contains `model.joblib` and `metrics.json`.

## 7. See the quality gate block a bad model

Run the workflow manually with weak settings: `n_estimators` = `10`, `learning_rate` = `0.01`, `max_depth` = `1`.

- `test` passes.
- `train` goes **red** with:
  ```
  ❌ Quality gate FAILED: accuracy 0.6316 < 0.9; malignant recall 0.0 < 0.85
  ```
- `upload` is **skipped**, and nothing new appears in the bucket.

**PR behavior:** open a pull request into `main`. `test` and `train` run, and `upload` is skipped.

## Troubleshooting

| Problem | Fix |
|---|---|
| Workflow doesn't show up in Actions | The file must be at `.github/workflows/` in the **repo root**. |
| `upload` fails at "Authenticate with GCP" | The `GCP_SA_KEY` secret is missing or isn't the full JSON contents. |
| `upload` fails with a 403 on `gcloud storage cp` | The service account lacks the Storage Object Admin role on the bucket. |
| `upload` fails with an empty bucket name (`gs:///models/...`) | The `GCS_BUCKET_NAME` variable isn't set. It goes under **Variables**, not Secrets. |
