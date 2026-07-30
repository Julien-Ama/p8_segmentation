from io import BytesIO

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from inference import load_segmentation_model, predict_mask
import traceback

app = FastAPI(
    title="Cityscapes Segmentation API",
    version="1.0"
)

# Le modèle est chargé une seule fois au démarrage
model = load_segmentation_model()


@app.get("/")
def root():
    return {
        "message": "API de segmentation Cityscapes"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.get("/model-info")
def model_info():
    return {
        "model": "cityscapes_segmentation_final.keras",
        "input_size": [256, 256, 3],
        "num_classes": 8
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed_types = {
        "image/png",
        "image/jpeg"
    }

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Format non pris en charge. Utilisez une image PNG ou JPEG."
        )

    try:
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=400,
            detail="Le fichier envoyé n'est pas une image valide."
        )
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Impossible de lire l'image : {error}"
        )

    try:
        mask = predict_mask(
            model=model,
            image=image
        )
        except Exception as error:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                f"Erreur pendant la prédiction : "
                f"{type(error).__name__}: {error}"
            )
        )

    output_buffer = BytesIO()

    mask_image.save(
        output_buffer,
        format="PNG"
    )

    output_buffer.seek(0)

    return StreamingResponse(
        output_buffer,
        media_type="image/png",
        headers={
            "Content-Disposition": "inline; filename=predicted_mask.png"
        }
    )

# Pourquoi mode="L" ?
# Le masque contient les numéros de classes :
# 0, 1, 2, 3, 4, 5, 6, 7
#
# Le mode L crée une image en niveaux de gris à un seul canal.
# Chaque pixel conserve donc directement son identifiant de classe.
#
# Cette image semblera presque noire lorsqu'on l'ouvre normalement,
# car les valeurs vont seulement de 0 à 7 sur une échelle allant de 0 à 255.
# C'est normal : nous ajouterons ensuite une palette de couleurs dans Streamlit.