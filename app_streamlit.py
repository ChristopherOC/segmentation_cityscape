import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(page_title="Segmentation Demo", layout="centered")

st.title("🧠 Test de ton modèle de segmentation")

# Upload image
uploaded_file = st.file_uploader("Choisir une image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Affichage image originale
    image = Image.open(uploaded_file).convert("RGB")
    st.subheader("Image originale")
    st.image(image, width=300)

    if st.button("Lancer la prédiction"):
        with st.spinner("Prédiction en cours..."):
            try:
                # Appel API
                files = {"file": uploaded_file.getvalue()}
                response = requests.post("https://cityscape-g0e4fddycydje0ad.francecentral-01.azurewebsites.net", files={"file": uploaded_file})

                if response.status_code == 200:
                    # Convertir réponse en image
                    mask = Image.open(io.BytesIO(response.content))

                    st.subheader("Masque de prédiction")
                    st.image(mask, width=True)

                else:
                    st.error(f"Erreur API : {response.status_code}")

            except Exception as e:
                st.error(f"Erreur : {e}")
