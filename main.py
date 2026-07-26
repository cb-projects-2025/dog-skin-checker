"""
Dog Skin Disease Prediction API (TFLite version)
--------------------------------------------------
Deploy on Render (https://render.com) -- no ngrok, no Colab session needed.

Local test:
    uvicorn main:app --reload --host 0.0.0.0 --port 8000
Then open http://127.0.0.1:8000 in a browser.
"""

import io
import logging

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dog-skin-api")

MODEL_PATH = "dog_skin_disease_model.tflite"
IMG_SIZE = 256
CLASS_NAMES = [
    "Dermatitis",
    "Fungal_infections",
    "Healthy",
    "Hypersensitivity",
    "demodicosis",
    "ringworm",
]

app = FastAPI(
    title="Dog Skin Disease Prediction API",
    description="Upload a photo of a dog's skin and get a predicted condition.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

interpreter = None
input_details = None
output_details = None


@app.on_event("startup")
async def load_model():
    global interpreter, input_details, output_details
    logger.info("Loading TFLite model from %s ...", MODEL_PATH)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    logger.info("Model loaded. Classes: %s", CLASS_NAMES)


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    image = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def run_inference(img_array: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    return output[0]


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": interpreter is not None}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    img_array = preprocess_image(contents)
    predictions = run_inference(img_array)

    predicted_idx = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[predicted_idx])

    all_scores = {
        CLASS_NAMES[i]: round(float(predictions[i]), 4) for i in range(len(CLASS_NAMES))
    }

    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": round(confidence, 4),
        "all_scores": all_scores,
    }


# Serves the upload page (static/index.html) at the root URL
app.mount("/", StaticFiles(directory="static", html=True), name="static")
