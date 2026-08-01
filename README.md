# Semantic Segmentation for Autonomous Driving

Semantic segmentation prototype developed as part of the OpenClassrooms Artificial Intelligence Engineer curriculum.

The objective of this project is to build, evaluate and deploy a semantic segmentation model capable of identifying the main components of an urban road scene using the Cityscapes dataset.

---

## Project Overview

This project covers the complete lifecycle of an AI application:

- Dataset preparation
- Image preprocessing
- Semantic segmentation model training
- Model comparison and optimization
- REST API deployment
- Web application deployment

The final solution is based on a **U-Net architecture with a VGG16 encoder**, exposed through **FastAPI** and demonstrated with **Streamlit**.

---

## Dataset

The project uses the **Cityscapes** dataset.

The original 34 semantic classes were grouped into 8 categories:

- Void
- Flat
- Construction
- Object
- Nature
- Sky
- Human
- Vehicle

Images are resized to **256 × 256** before training.

---

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Pillow
- FastAPI
- Streamlit
- Git & GitHub
- Render

---

## Project Structure

```
p8_segmentation/
│
├── api/
│   ├── model/
│   ├── inference.py
│   ├── main.py
│   ├── requirements.txt
│   └── .python-version
│
├── streamlit_app/
│   ├── demo_data/
│   ├── app.py
│   ├── config.py
│   ├── utils.py
│   └── requirements.txt
│
├── notebooks/
│
├── README.md
└── .gitignore
```

---

## Model

Architecture:

- U-Net
- VGG16 pretrained encoder
- Batch Normalization
- Dropout
- AdamW optimizer

Evaluation metrics:

- Accuracy
- Mean IoU
- Dice Score

Loss function:

- Sparse Categorical Crossentropy

---

## Deployment

The application is composed of two services:

### FastAPI

Receives an image, performs inference and returns the predicted segmentation mask.

Deployed on **Render**.

### Streamlit

Provides a simple user interface allowing users to:

- Select an image
- Compare the original image
- Display the ground truth mask
- Display the predicted segmentation

Deployed on **Streamlit Community Cloud**.

---

## Repository

```bash
git clone https://github.com/Julien-Ama/p8_segmentation.git
cd p8_segmentation
```

---

## Installation

### API

```bash
cd api

pip install -r requirements.txt

uvicorn main:app --reload
```

---

### Streamlit

```bash
cd streamlit_app

pip install -r requirements.txt

streamlit run app.py
```

---

## Results

The project demonstrates a complete semantic segmentation pipeline, from dataset preparation to cloud deployment.

Several improvements were evaluated during development, including:

- VGG16 transfer learning
- Batch Normalization
- Dropout
- Data Augmentation
- Hyperparameter optimization

The final model offers a good compromise between segmentation quality and inference performance.

---

## Author

Julien Ama

OpenClassrooms – Artificial Intelligence Engineer
