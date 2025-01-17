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
        🌱 Lettuce Health Classification 🌱
    </h3>
    """,
    unsafe_allow_html=True
)

# Instructions
st.write(
    "<p style='text-align: center; font-size: 14px;'>📸 Upload an image or use your camera to classify the lettuce's health.</p>",
    unsafe_allow_html=True
)

# Image input options: Upload or Camera
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
camera_image = st.camera_input("Take a photo")

if uploaded_file or camera_image:
    # Process the uploaded file or camera input
    if uploaded_file:
        image = Image.open(uploaded_file)
    else:
        # Convert camera image bytes to a PIL Image
        image = Image.open(camera_image)

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
hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    img {max-width: 100%; height: auto;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
