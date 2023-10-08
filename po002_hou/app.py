from fastapi import FastAPI, File, UploadFile, Form
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import io

app = FastAPI()
model = load_model('/Users/cassidyhilton/Documents/Vian_Notebooks_Pers/portfolio/po002_hou/src/housePrices.h5')

@app.post("/predict/")
async def predict(n_citi: float = Form(...), bed: float = Form(...), bath: float = Form(...), sqft: float = Form(...), file: UploadFile = File(...)):

    tabular_data = np.array([[n_citi, bed, bath, sqft]])
    citiM, bM, bathM, sqftM, priceM = 414, 12, 36.0, 17667, 200000
    tabular_data = tabular_data / np.array([citiM, bM, bathM, sqftM])  # Normalize like you did during training

    # Read and preprocess the image
    image_data = await file.read()
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), -1)
    image = cv2.resize(image, (64, 64))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict([tabular_data, image])

    return {"prediction": float(prediction[0]*priceM)}  # Convert prediction to float or JSON serializable format
