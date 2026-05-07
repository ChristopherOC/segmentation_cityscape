from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import io

from model import predict

app = FastAPI(title="Segmentation API")

@app.get("/")
def home():
    return {"message": "API segmentation OK"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    # Lire image
    image = Image.open(file.file).convert("RGB")

    # Prédiction
    result = predict(image)

    # Convertir en image PNG
    img_bytes = io.BytesIO()
    Image.fromarray(result).save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")
