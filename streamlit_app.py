import streamlit as st
from inference_sdk import InferenceHTTPClient
from PIL import Image
import io
import tempfile

# Initialize the InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://classify.roboflow.com",
    api_key="m9qYjCzZlc0zYhxYaIpa"
)

# Title of the app
st.title("Lettuce Health Classification Model")

# Instructions
st.write("Upload an image of a lettuce to classify its health.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Create a temporary file to save the uploaded image
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
            st.write("### Prediction Results:")
            
            # Display the top prediction
            top_prediction = result["top"]
            confidence = result["confidence"]
            st.write(f"**Top Class:** {top_prediction} with confidence of {confidence*100:.2f}%")

            # Display detailed predictions
            st.write("### All Predictions:")
            for prediction in result["predictions"]:
                class_name = prediction["class"]
                class_confidence = prediction["confidence"]
                st.write(f"**Class:** {class_name} - Confidence: {class_confidence*100:.2f}%")
            
            # Add a confidence bar for the top prediction
            st.write("### Confidence Visualization:")
            st.progress(int(confidence * 100))

        else:
            st.write("No result from the model.")
            
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
