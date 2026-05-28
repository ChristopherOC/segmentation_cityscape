import streamlit as st
import requests
from PIL import Image
import io

st.set_page_config(
    page_title="Segmentation Demo",
    layout="wide"
)

st.title("🧠 Démonstration de segmentation d'image")

# Upload image
uploaded_file = st.file_uploader(
    "Choisir une image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    # Lecture image
    image = Image.open(uploaded_file).convert("RGB")

    # Bouton prédiction
    if st.button("Lancer la prédiction"):

        with st.spinner("Prédiction en cours..."):

            try:

                # Remet le curseur au début du fichier
                uploaded_file.seek(0)

                # Format correct pour requests
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                # Appel API FastAPI
                response = requests.post(
                    "https://cityscape-g0e4fddycydje0ad.francecentral-01.azurewebsites.net",
                    files=files
                )

                # Vérification réponse
                if response.status_code == 200:

                    # Conversion du masque retourné
                    mask = Image.open(io.BytesIO(response.content))

                    st.success("Prédiction effectuée avec succès")

                    # Affichage côte à côte
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Image originale")
                        st.image(image, use_container_width=True)

                    with col2:
                        st.subheader("Masque prédit")
                        st.image(mask, use_container_width=True)

                else:
                    st.error(f"Erreur API : {response.status_code}")
                    st.text(response.text)

            except Exception as e:
                st.error(f"Erreur : {e}")
