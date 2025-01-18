import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile

# Inisialisasi InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="m9qYjCzZlc0zYhxYaIpa"
)

# Judul yang lebih kecil untuk kenyamanan di perangkat mobile
st.markdown(
    """
    <h3 style='text-align: center; font-size: 22px; margin-top: 0; color: #4CAF50;'>
        Cek Nutrisi Selada
    </h3>
    """,
    unsafe_allow_html=True
)

# Instruksi dengan ikon dan gaya teks
st.write(
    "<p style='text-align: left; font-size: 16px; color: #555;'>📸 Unggah gambar untuk cek kesehatan selada</p>",
    unsafe_allow_html=True
)
st.write(
    "<p style='text-align: left; font-size: 16px; color: #555;'>✅ Pastikan hanya satu selada dalam satu frame gambar</p>",
    unsafe_allow_html=True
)

# Input gambar: Unggah
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    # Proses file yang diunggah
    image = Image.open(uploaded_file)

    # Tampilkan gambar
    st.image(image, caption="Gambar yang Dipilih", use_container_width=True)

    # Buat file sementara untuk menyimpan gambar
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_name = temp_file.name
        image.save(temp_file, format="JPEG")
        temp_file.flush()

        # Lakukan inferensi menggunakan file gambar sementara
        st.write("Menganalisis kesehatan selada...")

        # Gunakan path file sementara dalam inferensi
        result = CLIENT.infer(temp_file_name, model_id="lettuce-health-classification/1")

        # Tampilkan hasil
        if result:
            st.markdown("<h4 style='color: #4CAF50;'>Hasil Prediksi:</h4>", unsafe_allow_html=True)
            
            # Tampilkan prediksi teratas
            top_prediction = result["top"]
            confidence = result["confidence"]
            st.write(
                f"**Kelas Teratas:** {top_prediction} dengan kepercayaan {confidence*100:.2f}%"
            )

            # Tampilkan prediksi lengkap
            st.write("### Semua Prediksi:")
            for prediction in result["predictions"]:
                class_name = prediction["class"]
                class_confidence = prediction["confidence"]
                st.write(
                    f"**Kelas:** {class_name} - Kepercayaan: {class_confidence*100:.2f}%"
                )

            # Tambahkan bar visualisasi kepercayaan untuk prediksi teratas
            st.write("### Visualisasi Kepercayaan:")
            st.progress(int(confidence * 100))
        else:
            st.write("Tidak ada hasil dari model.")

# Gaya kustom untuk kenyamanan di perangkat mobile dan padding
hide_streamlit_style = """
                <style>
                body {
                    padding: 25px;
                    font-family: 'Arial', sans-serif;
                    background-color: #f4f4f9;
                    color: #333;
                }
                div[data-testid="stToolbar"] {
                    visibility: hidden;
                    height: 0%;
                    position: fixed;
                }
                div[data-testid="stDecoration"] {
                    visibility: hidden;
                    height: 0%;
                    position: fixed;
                }
                div[data-testid="stStatusWidget"] {
                    visibility: hidden;
                    height: 0%;
                    position: fixed;
                }
                #MainMenu {
                    visibility: hidden;
                    height: 0%;
                }
                header {
                    visibility: hidden;
                    height: 0%;
                    position: fixed;
                }
                footer {
                    visibility: hidden;
                    height: 0%;
                    position: fixed;
                }
                .css-1v3fvcr {
                    margin-top: 16px;
                }
                .stButton>button {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border-radius: 8px;
                    padding: 10px;
                    width: 100%;
                }
                .stButton>button:hover {
                    background-color: #45a049;
                }
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
