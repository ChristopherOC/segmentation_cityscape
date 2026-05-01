import streamlit as st
from PIL import Image
import io
import requests

#Configuration de la page
st.set_page_config(
    page_title="Segmentation",
    page_icon="🔬",
    layout="centered",
)

#CSS personnalisé
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0e0e0e;
    color: #e8e8e8;
}

/* Header */
.header {
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 1.2rem;
    margin-bottom: 2rem;
}
.header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    color: #ffffff;
    margin: 0;
}
.header p {
    font-size: 0.82rem;
    color: #666;
    margin: 0.3rem 0 0 0;
    font-weight: 300;
    letter-spacing: 0.03em;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    background: #141414 !important;
    border: 1px dashed #333 !important;
    border-radius: 4px !important;
    padding: 1.5rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #555 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #555 !important;
    font-size: 0.85rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
}

/* Bouton */
.stButton > button {
    background: #e8e8e8 !important;
    color: #0e0e0e !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    padding: 0.55rem 1.8rem !important;
    transition: background 0.15s, transform 0.1s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: #ffffff !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Images */
[data-testid="stImage"] img {
    border-radius: 3px;
    border: 1px solid #1e1e1e;
}

/* Labels colonnes */
.col-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #444;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

/* Tag de statut */
.status-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    letter-spacing: 0.06em;
}
.status-ok   { background: #1a2e1a; color: #4caf50; }
.status-err  { background: #2e1a1a; color: #f44336; }

/* Divider */
hr {
    border: none;
    border-top: 1px solid #1e1e1e;
    margin: 2rem 0;
}

/* Cacher éléments Streamlit */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header">
    <h1>SEGMENTATION</h1>
    <p>Semantic segmentation · UNet · 8 classes</p>
</div>
""", unsafe_allow_html=True)

#Configuration API
API_URL = st.sidebar.text_input(
    "API endpoint",
    value="http://localhost:8000/predict",
    help="URL de l'endpoint /predict de votre API FastAPI"
)

st.sidebar.markdown("""
<div style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#444; margin-top:1rem;">
CLASSES<br><br>
<span style="color:#807d7d">█</span> 0 · Background<br>
<span style="color:#8040a0">█</span> 1 · Road<br>
<span style="color:#464646">█</span> 2 · Building<br>
<span style="color:#999999">█</span> 3 · Fence/Pole<br>
<span style="color:#4c7d23">█</span> 4 · Vegetation<br>
<span style="color:#4682b4">█</span> 5 · Sky<br>
<span style="color:#dc143c">█</span> 6 · Person<br>
<span style="color:#00008e">█</span> 7 · Vehicle
</div>
""", unsafe_allow_html=True)

# Upload
uploaded = st.file_uploader(
    "Déposer une image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    label_visibility="collapsed"
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2, gap="medium")
    with col1:
        st.markdown('<div class="col-label">INPUT</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)

    # Prédiction
    st.markdown("")
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        run = st.button("PREDICT", use_container_width=True)

    if run:
        with st.spinner(""):
            try:
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                buf.seek(0)

                resp = requests.post(
                    API_URL,
                    files={"file": ("image.jpg", buf, "image/jpeg")},
                    timeout=60,
                )
                resp.raise_for_status()

                mask = Image.open(io.BytesIO(resp.content))
                with col2:
                    st.markdown('<div class="col-label">MASK</div>', unsafe_allow_html=True)
                    st.image(mask, use_container_width=True)

                st.markdown(
                    '<div style="text-align:center; margin-top:1rem;">'
                    '<span class="status-tag status-ok">✓ DONE</span>'
                    '</div>',
                    unsafe_allow_html=True
                )

            except requests.exceptions.ConnectionError:
                st.markdown(
                    '<span class="status-tag status-err">✗ API unreachable — vérifiez l\'endpoint</span>',
                    unsafe_allow_html=True
                )
            except requests.exceptions.HTTPError as e:
                st.markdown(
                    f'<span class="status-tag status-err">✗ HTTP {e.response.status_code}</span>',
                    unsafe_allow_html=True
                )
            except Exception as e:
                st.markdown(
                    f'<span class="status-tag status-err">✗ Erreur : {e}</span>',
                    unsafe_allow_html=True
                )

else:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color:#2e2e2e;
                font-family:'IBM Plex Mono',monospace; font-size:0.75rem;
                letter-spacing:0.1em;">
        — aucune image sélectionnée —
    </div>
    """, unsafe_allow_html=True)