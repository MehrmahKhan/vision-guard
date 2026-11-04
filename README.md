# VisionGuard

![CI](https://github.com/MehrmahKhan/23L-0623_A2_Ops/actions/workflows/ci.yml/badge.svg)

A web service that judges whether an uploaded image is AI-generated or a real
photograph. TensorFlow/Keras for the model, MLflow on DagsHub for experiment
tracking, DVC for data versioning, Flask for the interface, Postgres for the
request trail, Docker Compose to run it all, and a cloud-hosted model API that
the local Flask app calls over the public internet.

MLOps Assignment 2.

## Links

| What | Where |
|---|---|
| GitHub | `https://github.com/MehrmahKhan/23L-0623_A2_Ops` |
| DagsHub | `https://dagshub.com/MehrmahKhan/23L-0623_A2_Ops` |
| MLflow | `https://dagshub.com/MehrmahKhan/23L-0623_A2_Ops.mlflow` |
| Model API | `https://<your-cloud-host>/predict` |
| Graded commit | `<fill this in once you tag the submission>` |

## Layout

```
src/            training, evaluation, tuning, MLflow and registry scripts
toy/            the toy labs (MLflow local, Keras Tuner)
app/            the Flask web app, inference module and database layer
model_api/      the small standalone model API that goes on the cloud
tests/          pytest suite that CI runs
Dockerfile      image for the web app
docker-compose.yml   web app plus Postgres
.github/workflows/   CI
```

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
copy .env.example .env           # then fill it in

python -m src.audit_dataset
python -m src.split_data
python -m src.train --model transfer --finetune --run-name run1
python -m app.app
```

Full step-by-step instructions, including DVC, DagsHub, Docker and deployment,
are in the runbook that accompanies this repository.

## Environment variables

See `.env.example`. Nothing secret is ever committed: `.env` is git-ignored and
GitHub Actions secrets live in the repository settings.
