"""Converts the champion Keras model into a lightweight format for deployment.

Full TensorFlow needs more memory than a free cloud tier provides, so the
serving container runs a small runtime instead. This script produces a TFLite
file, which is the smallest option and needs no extra packages to create.

Run from the project root:
    python convert_model.py
"""

import os

import tensorflow as tf

SOURCE = os.path.join("models", "champion.keras")
EXPORT_DIR = os.path.join("models", "champion_savedmodel")
TFLITE_OUT = os.path.join("models", "champion.tflite")


def main():
    if not os.path.exists(SOURCE):
        raise SystemExit("Missing " + SOURCE)

    print("loading " + SOURCE)
    model = tf.keras.models.load_model(SOURCE)
    print("input shape: " + str(model.input_shape))

    print("exporting to SavedModel ...")
    model.export(EXPORT_DIR)

    print("converting to TFLite ...")
    converter = tf.lite.TFLiteConverter.from_saved_model(EXPORT_DIR)
    # No quantization. Dynamic-range quantization shrinks the file further but
    # produced a FULLY_CONNECTED layer the XNNPACK delegate could not prepare,
    # and the saving is not needed: the memory win comes from not installing
    # TensorFlow in the serving image at all, not from the file size.
    tflite_bytes = converter.convert()

    handle = open(TFLITE_OUT, "wb")
    handle.write(tflite_bytes)
    handle.close()

    original_mb = os.path.getsize(SOURCE) / (1024.0 * 1024.0)
    converted_mb = os.path.getsize(TFLITE_OUT) / (1024.0 * 1024.0)

    print("")
    print("wrote " + TFLITE_OUT)
    print("original  : " + str(round(original_mb, 2)) + " MB")
    print("converted : " + str(round(converted_mb, 2)) + " MB")
    print("reduction : " + str(round(100.0 * (1 - converted_mb / original_mb), 1)) + "%")

    # Sanity check: the converted model must agree with the original.
    print("")
    print("verifying the converted model agrees with the original ...")
    import numpy as np

    size = model.input_shape[1]
    rng = np.random.default_rng(42)
    sample = rng.random((1, size, size, 3)).astype("float32") * 2.0 - 1.0

    keras_output = float(model.predict(sample, verbose=0)[0][0])

    interpreter = tf.lite.Interpreter(model_path=TFLITE_OUT)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], sample)
    interpreter.invoke()
    tflite_output = float(interpreter.get_tensor(output_details[0]["index"])[0][0])

    print("keras  says: " + str(round(keras_output, 6)))
    print("tflite says: " + str(round(tflite_output, 6)))
    print("difference : " + str(round(abs(keras_output - tflite_output), 6)))

    if abs(keras_output - tflite_output) < 0.05:
        print("")
        print("OK, the converted model matches closely enough to deploy.")
    else:
        print("")
        print("WARNING: outputs differ more than expected, check before deploying.")


if __name__ == "__main__":
    main()
