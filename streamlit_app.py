import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import tempfile

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="m9qYjCzZlc0zYhxYaIpa"
)

# Scaled-down title for mobile-friendliness
st.markdown(
    """
    <h3 style='text-align: center; font-size: 18px; margin-top: 0;'>
        Cek Nutrisi Selada
    </h3>
    """,
    unsafe_allow_html=True
)

# Instructions (left aligned)
st.write(
    "<p style='text-align: left; font-size: 14px;'>📸 Upload gambar untuk cek kesehatan selada</p>",
    unsafe_allow_html=True
)
st.write(
    "<p style='text-align: left; font-size: 14px;'>✅ Pastikan hanya satu selada dalam satu frame gambar</p>",
    unsafe_allow_html=True
)

# Image input: Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    # Process the uploaded file
    image = Image.open(uploaded_file)

    # Display the image
    st.image(image, caption="Selected Image", use_container_width=True)

    # Create a temporary file to save the image
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file_name = temp_file.name
        image.save(temp_file, format="JPEG")
        temp_file.flush()

        # Perform inference using the temporary image file
        st.write("Classifying the health of the lettuce...")

        # Use the temporary file path in the inference
        result = CLIENT.infer(temp_file_name, model_id="lettuce-health-classification/1")

        # Display the result
        if result:
            st.markdown("<h4>Prediction Results:</h4>", unsafe_allow_html=True)
            
            # Display the top prediction
            top_prediction = result["top"]
            confidence = result["confidence"]
            st.write(
                f"**Top Class:** {top_prediction} with confidence of {confidence*100:.2f}%"
            )

            # Display detailed predictions
            st.write("### All Predictions:")
            for prediction in result["predictions"]:
                class_name = prediction["class"]
                class_confidence = prediction["confidence"]
                st.write(
                    f"**Class:** {class_name} - Confidence: {class_confidence*100:.2f}%"
                )

            # Add a confidence bar for the top prediction
            st.write("### Confidence Visualization:")
            st.progress(int(confidence * 100))
        else:
            st.write("No result from the model.")

# Custom styles for mobile-friendliness
hide_streamlit_style = """
                <style>
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
                </style>
                """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
