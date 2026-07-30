import requests
import streamlit as st

from config import (
    MODEL_INPUT_SIZE,
    MODEL_NAME,
    NUM_CLASSES,
)
from utils import (
    colorize_mask,
    get_available_image_ids,
    get_image_path,
    get_present_classes,
    load_ground_truth_mask,
    load_original_image,
    map_cityscapes_to_8_classes,
    request_prediction,
)
from PIL import Image

st.set_page_config(
    page_title="Segmentation Cityscapes",
    page_icon="🚗",
    layout="wide",
)


st.title("Segmentation sémantique Cityscapes")

st.write(
    "Sélectionnez une image du jeu de démonstration "
    "pour comparer le masque réel au masque prédit."
)


image_ids = get_available_image_ids()

if not image_ids:
    st.error(
        "Aucune paire image/masque valide n'a été trouvée "
        "dans demo_data."
    )

    st.stop()


selected_image_id = st.selectbox(
    "Choisir une image",
    options=[None] + image_ids,
    format_func=lambda value: (
        "Sélectionnez une image..."
        if value is None
        else value
    ),
)

if selected_image_id is None:
    st.info(
        "Choisissez une image dans la liste pour lancer la segmentation."
    )
    st.stop()


try:
    original_image = load_original_image(
        selected_image_id
    )

    ground_truth_mask = load_ground_truth_mask(
        selected_image_id
    )

    ground_truth_array = map_cityscapes_to_8_classes(
        ground_truth_mask
    )

    ground_truth_color = colorize_mask(
        ground_truth_array
    )

    image_path = get_image_path(
        selected_image_id
    )

    with st.spinner("Prédiction en cours..."):
        predicted_array, elapsed_time = request_prediction(
            image_path
        )

    predicted_color = colorize_mask(
        predicted_array
    )

    DISPLAY_SIZE = (512, 256)

    original_display = original_image.resize(
        DISPLAY_SIZE,
        Image.Resampling.LANCZOS
    )

    ground_truth_display = ground_truth_color.resize(
        DISPLAY_SIZE,
        Image.Resampling.NEAREST
    )

    predicted_display = predicted_color.resize(
        DISPLAY_SIZE,
        Image.Resampling.NEAREST
    )


    column_original, column_truth, column_prediction = (
        st.columns(3)
    )

    with column_original:
        st.subheader("Image originale")
        st.image(
            original_display,
            use_column_width=True,
        )

    with column_truth:
        st.subheader("Masque réel")
        st.image(
            ground_truth_display,
            use_column_width=True,
        )

    with column_prediction:
        st.subheader("Masque prédit")
        st.image(
            predicted_display,
            use_column_width=True,
        )

    st.divider()

    st.subheader("Informations")

    info_model, info_size, info_time, info_classes = (
        st.columns(4)
    )

    with info_model:
        st.metric(
            "Modèle",
            MODEL_NAME,
        )

    with info_size:
        st.metric(
            "Entrée du modèle",
            MODEL_INPUT_SIZE,
        )

    with info_time:
        st.metric(
            "Temps de prédiction",
            f"{elapsed_time:.3f} s",
        )

    with info_classes:
        st.metric(
            "Nombre de classes",
            NUM_CLASSES,
        )

    present_classes = get_present_classes(
        predicted_array
    )

    with st.expander(
        "Classes détectées dans le masque prédit"
    ):
        for class_name in present_classes:
            st.write(f"• {class_name}")

except requests.ConnectionError:
    st.error(
        "Impossible de contacter l'API FastAPI. "
        "Vérifiez qu'elle fonctionne sur "
        "http://127.0.0.1:8000."
    )

except requests.Timeout:
    st.error(
        "La requête vers l'API a dépassé "
        "le délai autorisé."
    )

except Exception as error:
    st.error(
        f"Une erreur est survenue : {error}"
    )