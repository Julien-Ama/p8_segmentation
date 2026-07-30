from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent

IMAGES_DIR = BASE_DIR / "demo_data" / "images"
MASKS_DIR = BASE_DIR / "demo_data" / "masks"

API_URL = "https://p8-segmentation-api.onrender.com"

MODEL_NAME = "U-Net + VGG16"
MODEL_INPUT_SIZE = "256 × 256"
NUM_CLASSES = 8


CLASS_NAMES = {
    0: "Void",
    1: "Flat",
    2: "Construction",
    3: "Object",
    4: "Nature",
    5: "Sky",
    6: "Human",
    7: "Vehicle",
}


COLOR_PALETTE = np.array(
    [
        [0, 0, 0],        # 0 - Void
        [128, 64, 128],   # 1 - Flat
        [70, 70, 70],     # 2 - Construction
        [153, 153, 153],  # 3 - Object
        [107, 142, 35],   # 4 - Nature
        [70, 130, 180],   # 5 - Sky
        [220, 20, 60],    # 6 - Human
        [0, 0, 142],      # 7 - Vehicle
    ],
    dtype=np.uint8,
)