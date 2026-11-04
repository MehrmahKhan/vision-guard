"""Bonus (+1) - Grad-CAM heat maps for 4 correct and 4 wrong predictions.

Run:
    python -m src.gradcam --model models/champion.keras
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

from src import config


def find_last_conv_layer(model):
    """Walks the model backwards for the last 4D output, including inside a
    nested backbone such as MobileNetV2."""
    layers = list(model.layers)
    index = len(layers) - 1
    while index >= 0:
        layer = layers[index]
        if hasattr(layer, "layers"):
            inner_index = len(layer.layers) - 1
            while inner_index >= 0:
                inner = layer.layers[inner_index]
                if len(inner.output.shape) == 4:
                    return layer, inner.name
                inner_index = inner_index - 1
        else:
            if len(layer.output.shape) == 4:
                return None, layer.name
        index = index - 1
    raise SystemExit("No convolutional layer found.")


def gradcam_heatmap(model, image_batch):
    container, layer_name = find_last_conv_layer(model)

    if container is None:
        conv_layer = model.get_layer(layer_name)
        grad_model = tf.keras.Model(model.inputs, [conv_layer.output, model.output])
    else:
        inner_model = tf.keras.Model(
            container.inputs, container.get_layer(layer_name).output
        )
        feature_maps = inner_model(model.inputs)
        grad_model = tf.keras.Model(model.inputs, [feature_maps, model.output])

    with tf.GradientTape() as tape:
        conv_output, prediction = grad_model(image_batch)
        target = prediction[:, 0]

    gradients = tape.gradient(target, conv_output)
    pooled = tf.reduce_mean(gradients, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)
    if float(maximum) > 0:
        heatmap = heatmap / maximum
    return heatmap.numpy(), float(prediction[0][0])


def load_batch(path, img_size):
    image = Image.open(path).convert("RGB").resize((img_size, img_size))
    array = np.array(image).astype("float32")
    image.close()
    display = array.astype("uint8")
    array = (array / 127.5) - 1.0
    return np.expand_dims(array, axis=0), display


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.path.join("models", "champion.keras"))
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    parser.add_argument("--threshold", type=float, default=config.DECISION_THRESHOLD)
    args = parser.parse_args()

    config.ensure_dirs()
    model = tf.keras.models.load_model(args.model)
    frame = pd.read_csv(os.path.join(config.SPLIT_DIR, "test.csv"))
    frame = frame.sample(frac=1.0, random_state=config.SEED)

    correct = []
    wrong = []
    position = 0
    while position < len(frame) and (len(correct) < 4 or len(wrong) < 4):
        row = frame.iloc[position]
        batch, display = load_batch(row["path"], args.img_size)
        heatmap, probability = gradcam_heatmap(model, batch)
        if probability >= args.threshold:
            predicted = 1
        else:
            predicted = 0

        record = {}
        record["display"] = display
        record["heatmap"] = heatmap
        record["probability"] = probability
        record["true"] = int(row["label"])
        record["pred"] = predicted

        if predicted == int(row["label"]):
            if len(correct) < 4:
                correct.append(record)
        else:
            if len(wrong) < 4:
                wrong.append(record)
        position = position + 1

    selected = correct + wrong
    figure, axes = plt.subplots(2, 4, figsize=(13, 7))
    index = 0
    row_index = 0
    while row_index < 2:
        column = 0
        while column < 4:
            axis = axes[row_index][column]
            axis.set_xticks([])
            axis.set_yticks([])
            if index < len(selected):
                record = selected[index]
                axis.imshow(record["display"])
                resized = np.array(
                    Image.fromarray((record["heatmap"] * 255).astype("uint8")).resize(
                        (args.img_size, args.img_size)
                    )
                )
                axis.imshow(resized, cmap="jet", alpha=0.45)
                axis.set_title(
                    "true " + config.CLASS_NAMES[record["true"]]
                    + " / pred " + config.CLASS_NAMES[record["pred"]]
                    + "\np=" + str(round(record["probability"], 3)),
                    fontsize=8,
                )
            index = index + 1
            column = column + 1
        row_index = row_index + 1

    figure.suptitle("Grad-CAM, top row correct, bottom row wrong")
    figure.tight_layout()
    output = os.path.join(config.ARTIFACT_DIR, "gradcam.png")
    figure.savefig(output, dpi=130)
    plt.close(figure)
    print("wrote " + output)


if __name__ == "__main__":
    main()
