from pathlib import Path

import requests
from PIL import Image
from io import BytesIO


BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "test_image.png"
OUTPUT_PATH = BASE_DIR / "predicted_mask.png"

API_URL = "http://127.0.0.1:8000/predict"


with IMAGE_PATH.open("rb") as image_file:
    response = requests.post(
        API_URL,
        files={
            "file": (
                IMAGE_PATH.name,
                image_file,
                "image/png"
            )
        },
        timeout=120
    )

response.raise_for_status()

mask_image = Image.open(
    BytesIO(response.content)
)

mask_image.save(OUTPUT_PATH)

print("Statut API :", response.status_code)
print("Format reçu :", mask_image.format)
print("Taille du masque :", mask_image.size)
print("Mode du masque :", mask_image.mode)
print("Masque enregistré dans :", OUTPUT_PATH)