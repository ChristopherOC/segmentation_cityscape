import torch
from PIL import Image
import numpy as np
import torchvision.transforms as T

import segmentation_models_pytorch as smp
import os
from azure.storage.blob import BlobClient

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformation
img_transform = T.Compose([
    T.Resize((256, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# Couleurs
SUPERCAT_COLORS = {
    0: (0,0,0),
    1: (128,64,128),
    2: (70,70,70),
    3: (153,153,153),
    4: (107,142,35),
    5: (70,130,180),
    6: (220,20,60),
    7: (0,0,142),
}

MODEL_PATH = "unet_u_model.pth"

# Variables Azure
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "unetmodel"
BLOB_NAME = "unet_u_model.pth"


def download_model():

    # Si le modèle existe déjàne rien faire
    if os.path.exists(MODEL_PATH):
        print("Modèle déjà présent")
        return

    print("Téléchargement du modèle depuis Azure Blob Storage...")

    blob_client = BlobClient.from_connection_string(
        conn_str=AZURE_CONNECTION_STRING,
        container_name=CONTAINER_NAME,
        blob_name=BLOB_NAME
    )

    with open(MODEL_PATH, "wb") as f:
        data = blob_client.download_blob()
        f.write(data.readall())

    print("Téléchargement terminé")

def load_model():
    model = smp.Unet(
        encoder_name="efficientnet-b4",
        encoder_weights=None,
        in_channels=3,
        classes=8
    )

    checkpoint = torch.load("unet_u_model.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()
    return model

download_model()
model = load_model()


def seg_map_to_rgb(seg_map):
    h, w = seg_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in SUPERCAT_COLORS.items():
        rgb[seg_map == idx] = color
    return rgb

def predict(image: Image.Image):
    img_tensor = img_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_tensor)
        seg_map = logits.argmax(dim=1).squeeze().cpu().numpy()

    return seg_map_to_rgb(seg_map)