"""Tasks 2.2 and 2.3 - the two model definitions.

build_baseline_cnn  : a small convolutional network written from scratch
build_transfer_model: a frozen keras.applications backbone plus our own head

Both end in a single sigmoid unit, so the loss is binary cross-entropy and the
output reads directly as "probability that this image is AI-generated".
"""

import tensorflow as tf

from src import config


def metric_list():
    """The metrics we track every epoch.

    Accuracy alone hides the behaviour we care about, so precision, recall and
    AUC come along. AUC is threshold-free, which matters because we pick the
    decision threshold later from the validation set.
    """
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]


def build_baseline_cnn(img_size=None, dropout=0.4, learning_rate=1e-3):
    """Small CNN, four convolution blocks.

    Layer by layer:
      Rescaling        - maps 0..255 to 0..1 so gradients stay in a sane range
      Conv2D           - learns local filters; the filter count doubles as the
                         spatial size halves, the standard pyramid
      BatchNormalization - keeps each layer's input distribution steady, which
                         lets us use a larger learning rate and trains faster
      MaxPooling2D     - halves height and width, so later filters see a wider
                         area of the original image for the same kernel size
      GlobalAveragePooling2D - collapses the feature map to one number per
                         channel. Cheaper and far less prone to overfitting
                         than Flatten + a big Dense layer
      Dropout          - randomly zeroes activations during training so the
                         head cannot rely on any single feature
      Dense(1, sigmoid) - one probability out
    """
    if img_size is None:
        img_size = config.IMG_SIZE

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    filters = [32, 64, 128, 128]
    block_index = 0
    while block_index < len(filters):
        count = filters[block_index]
        name_suffix = str(block_index + 1)
        x = tf.keras.layers.Conv2D(
            count, 3, padding="same", activation="relu", name="conv" + name_suffix
        )(x)
        x = tf.keras.layers.BatchNormalization(name="bn" + name_suffix)(x)
        x = tf.keras.layers.MaxPooling2D(2, name="pool" + name_suffix)(x)
        block_index = block_index + 1

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="verdict")(x)

    model = tf.keras.Model(inputs, outputs, name="baseline_cnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=metric_list(),
    )
    return model


def get_backbone(name, img_size):
    """Returns (backbone, preprocess_fn) for one of three allowed backbones.

    The preprocessing function matters: MobileNetV2 was trained on inputs scaled
    to -1..1, ResNet50 on Caffe-style mean-subtracted BGR, EfficientNetB0 on raw
    0..255. Feeding the wrong range destroys the pretrained features, so we
    always use the function that ships with the backbone instead of guessing.
    """
    shape = (img_size, img_size, 3)
    lowered = name.lower()

    if lowered == "mobilenetv2":
        backbone = tf.keras.applications.MobileNetV2(
            input_shape=shape, include_top=False, weights="imagenet"
        )
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    elif lowered == "efficientnetb0":
        backbone = tf.keras.applications.EfficientNetB0(
            input_shape=shape, include_top=False, weights="imagenet"
        )
        preprocess = tf.keras.applications.efficientnet.preprocess_input
    elif lowered == "resnet50":
        backbone = tf.keras.applications.ResNet50(
            input_shape=shape, include_top=False, weights="imagenet"
        )
        preprocess = tf.keras.applications.resnet50.preprocess_input
    else:
        raise ValueError("Unknown backbone: " + str(name))

    return backbone, preprocess


def build_transfer_model(
    backbone_name="MobileNetV2", img_size=None, dropout=0.3, learning_rate=1e-3
):
    """Stage 1: the backbone is frozen, only the new head learns."""
    if img_size is None:
        img_size = config.IMG_SIZE

    backbone, preprocess = get_backbone(backbone_name, img_size)
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="verdict")(x)

    model = tf.keras.Model(inputs, outputs, name="transfer_" + backbone_name.lower())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=metric_list(),
    )
    return model, preprocess, backbone


def unfreeze_last_layers(model, backbone, how_many=20, learning_rate=1e-5):
    """Stage 2: fine-tuning.

    Only the last `how_many` layers of the backbone become trainable, and the
    learning rate drops by two orders of magnitude. A normal learning rate here
    would wipe out the pretrained weights in the first few steps, which is the
    classic fine-tuning mistake.

    BatchNormalization layers stay frozen on purpose: their running statistics
    come from ImageNet and updating them on a small batch destabilises training.
    """
    backbone.trainable = True

    total = len(backbone.layers)
    cutoff = total - how_many
    index = 0
    while index < total:
        layer = backbone.layers[index]
        if index < cutoff:
            layer.trainable = False
        else:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True
        index = index + 1

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=metric_list(),
    )
    return model


def trainable_layer_names(model):
    names = []
    for layer in model.layers:
        if hasattr(layer, "layers"):
            for inner in layer.layers:
                if inner.trainable:
                    names.append(layer.name + "/" + inner.name)
        else:
            if layer.trainable and len(layer.trainable_weights) > 0:
                names.append(layer.name)
    return names
