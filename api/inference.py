from pathlib import Path

import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from PIL import Image


IMG_SIZE = (256, 256)
NUM_CLASSES = 8

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "model"
    / "cityscapes_segmentation_final.keras"
)


def load_segmentation_model() -> tf.keras.Model:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    return np.expand_dims(
        image_array,
        axis=0
    )


# def predict_mask(
#     model: tf.keras.Model,
#     image: Image.Image
# ) -> np.ndarray:
#     input_batch = preprocess_image(image)
#
#     prediction = model.predict(
#         input_batch,
#         verbose=0
#     )
#
#     mask = np.argmax(
#         prediction[0],
#         axis=-1
#     )
#
#     return mask.astype(np.uint8)
def predict_mask(
    model: tf.keras.Model,
    image: Image.Image
) -> np.ndarray:
    input_batch = preprocess_image(image)

    prediction = model(
        input_batch,
        training=False
    )

    prediction = prediction.numpy()

    mask = np.argmax(
        prediction[0],
        axis=-1
    )

    return mask.astype(np.uint8)