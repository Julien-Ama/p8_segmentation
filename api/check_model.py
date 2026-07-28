from pathlib import Path
from PIL import Image

from inference import load_segmentation_model, predict_mask


BASE_DIR = Path(__file__).resolve().parent
IMAGE_PATH = BASE_DIR / "test_image.png"

model = load_segmentation_model()
image = Image.open(IMAGE_PATH)

mask = predict_mask(model, image)

print("Shape du masque :", mask.shape)
print("Classes prédites :", sorted(set(mask.flatten())))