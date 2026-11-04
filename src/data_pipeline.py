"""Task 2.1 - the tf.data input pipeline.

Stages, in order:
  1. from_tensor_slices over the (path, label) lists read from the split CSVs
  2. shuffle           - training only, buffer is the whole split
  3. map(load_image)   - read bytes, decode, resize to IMG_SIZE x IMG_SIZE
  4. batch
  5. map(augment)      - training only, applied per batch on the GPU/CPU graph
  6. prefetch          - overlaps data preparation with model execution

Resizing to a square ignores the original aspect ratio, so a tall portrait gets
squashed. That is deliberate and consistent: every image, at training and at
serving time, gets the same distortion, so the model never sees a shape it was
not trained on. The alternative (resize_with_pad) keeps the ratio but adds black
bars, which become a feature the model can latch onto.
"""

import os

import pandas as pd
import tensorflow as tf

from src import config

AUTOTUNE = tf.data.AUTOTUNE


def read_split(name):
    path = os.path.join(config.SPLIT_DIR, name + ".csv")
    if not os.path.exists(path):
        raise SystemExit("Missing " + path + ". Run `python -m src.split_data` first.")
    frame = pd.read_csv(path)
    return frame["path"].tolist(), frame["label"].tolist()


def load_image(path, label, img_size):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, [img_size, img_size])
    image = tf.cast(image, tf.float32)
    image.set_shape([img_size, img_size, 3])
    return image, tf.cast(label, tf.float32)


def build_augmenter():
    """Random augmentation, training split only.

    Flips and small rotations/zooms/contrast shifts teach the model that the
    verdict must not depend on orientation or exposure. Nothing here changes
    colour balance aggressively, because colour artefacts are one of the real
    signals that separate AI images from photographs.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=config.SEED),
            tf.keras.layers.RandomRotation(0.05, seed=config.SEED),
            tf.keras.layers.RandomZoom(0.10, seed=config.SEED),
            tf.keras.layers.RandomContrast(0.10, seed=config.SEED),
        ],
        name="augmenter",
    )


def make_dataset(
    split_name,
    img_size=None,
    batch_size=None,
    augment=False,
    shuffle=False,
    preprocess_fn=None,
):
    if img_size is None:
        img_size = config.IMG_SIZE
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    paths, labels = read_split(split_name)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths), seed=config.SEED, reshuffle_each_iteration=True
        )

    def mapper(path, label):
        return load_image(path, label, img_size)

    dataset = dataset.map(mapper, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(batch_size)

    if augment:
        augmenter = build_augmenter()

        def augment_batch(images, batch_labels):
            return augmenter(images, training=True), batch_labels

        dataset = dataset.map(augment_batch, num_parallel_calls=AUTOTUNE)

    if preprocess_fn is not None:

        def preprocess_batch(images, batch_labels):
            return preprocess_fn(images), batch_labels

        dataset = dataset.map(preprocess_batch, num_parallel_calls=AUTOTUNE)

    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


def save_augmented_batch_figure(output_path=None):
    """Writes the 'one augmented batch' figure that sub-task 2.1 asks for."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if output_path is None:
        output_path = os.path.join(config.ARTIFACT_DIR, "augmented_batch.png")

    dataset = make_dataset("train", augment=True, shuffle=True)
    images, labels = next(iter(dataset))

    figure, axes = plt.subplots(3, 4, figsize=(10, 7.5))
    index = 0
    row = 0
    while row < 3:
        column = 0
        while column < 4:
            axis = axes[row][column]
            if index < images.shape[0]:
                axis.imshow(images[index].numpy().astype("uint8"))
                label_index = int(labels[index].numpy())
                axis.set_title(config.CLASS_NAMES[label_index], fontsize=9)
            axis.set_xticks([])
            axis.set_yticks([])
            index = index + 1
            column = column + 1
        row = row + 1

    figure.suptitle("One augmented training batch (first 12 images)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=130)
    plt.close(figure)
    print("wrote " + output_path)
    return output_path


if __name__ == "__main__":
    config.ensure_dirs()
    save_augmented_batch_figure()
