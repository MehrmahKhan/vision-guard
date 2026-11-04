"""Task 3.6 - the separate small script that loads the model by alias and
classifies one image.

Run:
    python -m src.load_champion --image path/to/some_image.jpg
    python -m src.load_champion --image path/to/x.jpg --download-to models/champion.keras
"""

import argparse
import os
import shutil

import mlflow
import numpy as np
from PIL import Image

from src import config

MODEL_NAME = "VisionGuard"


def preprocess_image(path, img_size):
    image = Image.open(path).convert("RGB").resize((img_size, img_size))
    array = np.array(image).astype("float32")
    image.close()
    # MobileNetV2 expects -1..1
    array = (array / 127.5) - 1.0
    return np.expand_dims(array, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--alias", default="champion")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    parser.add_argument(
        "--download-to",
        default=None,
        help="also copy the resolved .keras file here, for the Flask app to serve",
    )
    args = parser.parse_args()

    if config.MLFLOW_TRACKING_URI != "":
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    model_uri = "models:/" + args.model_name + "@" + args.alias
    print("loading " + model_uri)
    model = mlflow.tensorflow.load_model(model_uri)
    print("loaded")

    batch = preprocess_image(args.image, args.img_size)
    probability = float(model.predict(batch, verbose=0)[0][0])

    if probability >= config.DECISION_THRESHOLD:
        verdict = "AI"
        confidence = probability
    else:
        verdict = "REAL"
        confidence = 1.0 - probability

    print("")
    print("image      : " + args.image)
    print("p(AI)      : " + str(round(probability, 4)))
    print("threshold  : " + str(config.DECISION_THRESHOLD))
    print("verdict    : " + verdict)
    print("confidence : " + str(round(confidence * 100.0, 2)) + "%")

    if args.download_to is not None:
        local_path = mlflow.artifacts.download_artifacts(model_uri)
        os.makedirs(os.path.dirname(args.download_to), exist_ok=True)
        found = None
        for current_dir, _subdirs, files in os.walk(local_path):
            for file_name in files:
                if file_name.endswith(".keras") or file_name.endswith(".h5"):
                    found = os.path.join(current_dir, file_name)
                    break
            if found is not None:
                break
        if found is None:
            model.save(args.download_to)
            print("re-saved the loaded model to " + args.download_to)
        else:
            shutil.copy(found, args.download_to)
            print("copied " + found + " to " + args.download_to)


if __name__ == "__main__":
    main()
