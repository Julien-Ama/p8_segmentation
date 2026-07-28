from io import BytesIO
from pathlib import Path
from time import perf_counter

import numpy as np
import requests
from PIL import Image

from config import (
    API_URL,
    CLASS_NAMES,
    COLOR_PALETTE,
    IMAGES_DIR,
    MASKS_DIR,
)


def get_available_image_ids() -> list[str]:
    """
    Récupère les identifiants des images disponibles
    dans le dossier de démonstration.
    """
    image_ids = []

    for image_path in sorted(
        IMAGES_DIR.glob("*_leftImg8bit.png")
    ):
        image_id = image_path.name.replace(
            "_leftImg8bit.png",
            "",
        )

        mask_path = (
            MASKS_DIR
            / f"{image_id}_gtFine_labelIds.png"
        )

        if mask_path.exists():
            image_ids.append(image_id)

    return image_ids


def get_image_path(image_id: str) -> Path:
    return (
        IMAGES_DIR
        / f"{image_id}_leftImg8bit.png"
    )


def get_mask_path(image_id: str) -> Path:
    return (
        MASKS_DIR
        / f"{image_id}_gtFine_labelIds.png"
    )


def load_original_image(image_id: str) -> Image.Image:
    image_path = get_image_path(image_id)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image introuvable : {image_path}"
        )

    return Image.open(image_path).convert("RGB")


def load_ground_truth_mask(image_id: str) -> Image.Image:
    mask_path = get_mask_path(image_id)

    if not mask_path.exists():
        raise FileNotFoundError(
            f"Masque introuvable : {mask_path}"
        )

    return Image.open(mask_path)


def map_cityscapes_to_8_classes(
    mask: Image.Image
) -> np.ndarray:
    """
    Convertit le masque Cityscapes labelIds
    vers les 8 super-classes du projet.
    """
    mask_array = np.asarray(
        mask,
        dtype=np.uint8,
    )

    mapped_mask = np.zeros_like(
        mask_array,
        dtype=np.uint8,
    )

    mapped_mask[
        (mask_array >= 0)
        & (mask_array <= 6)
    ] = 0

    mapped_mask[
        (mask_array >= 7)
        & (mask_array <= 10)
    ] = 1

    mapped_mask[
        (mask_array >= 11)
        & (mask_array <= 16)
    ] = 2

    mapped_mask[
        (mask_array >= 17)
        & (mask_array <= 20)
    ] = 3

    mapped_mask[
        (mask_array >= 21)
        & (mask_array <= 22)
    ] = 4

    mapped_mask[
        mask_array == 23
    ] = 5

    mapped_mask[
        (mask_array >= 24)
        & (mask_array <= 25)
    ] = 6

    mapped_mask[
        (mask_array >= 26)
        & (mask_array <= 33)
    ] = 7

    return mapped_mask


def colorize_mask(
    mask_array: np.ndarray
) -> Image.Image:
    """
    Convertit un masque 0 à 7 en image RGB.
    """
    mask_array = np.asarray(
        mask_array,
        dtype=np.uint8,
    )

    if mask_array.ndim == 3:
        mask_array = mask_array[:, :, 0]

    if mask_array.max() >= len(COLOR_PALETTE):
        raise ValueError(
            "Le masque contient une classe invalide."
        )

    color_mask = COLOR_PALETTE[mask_array]

    return Image.fromarray(
        color_mask,
        mode="RGB",
    )


def request_prediction(
    image_path: Path,
) -> tuple[np.ndarray, float]:
    """
    Envoie une image à l'API et retourne :
    - le masque brut
    - le temps total de requête
    """
    start_time = perf_counter()

    with image_path.open("rb") as image_file:
        response = requests.post(
            API_URL,
            files={
                "file": (
                    image_path.name,
                    image_file,
                    "image/png",
                )
            },
            timeout=120,
        )

    elapsed_time = perf_counter() - start_time

    if response.status_code != 200:
        try:
            detail = response.json().get(
                "detail",
                "Erreur inconnue",
            )
        except ValueError:
            detail = response.text

        raise RuntimeError(
            f"Erreur API {response.status_code} : {detail}"
        )

    predicted_mask = Image.open(
        BytesIO(response.content)
    )

    predicted_array = np.asarray(
        predicted_mask,
        dtype=np.uint8,
    )

    return predicted_array, elapsed_time


def get_present_classes(
    mask_array: np.ndarray
) -> list[str]:
    """
    Retourne les noms des classes présentes
    dans le masque.
    """
    class_ids = np.unique(
        mask_array
    ).tolist()

    return [
        CLASS_NAMES[class_id]
        for class_id in class_ids
        if class_id in CLASS_NAMES
    ]