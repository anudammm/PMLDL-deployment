import streamlit as st
import numpy as np
import requests
from PIL import Image
from streamlit_drawable_canvas import st_canvas
# Title
st.title("Draw a digit and get the prediction")

# Drawing canvas
st.sidebar.header("Canvas Settings")
brush_size = st.sidebar.slider("Brush size:", 1, 40, 15)
stroke_color = st.sidebar.color_picker("Stroke color:", "#000000")
bg_color = st.sidebar.color_picker("Background color:", "#FFFFFF")

canvas_result = st_canvas(
    fill_color=bg_color,
    stroke_width=brush_size,
    stroke_color=stroke_color,
    background_color="#FFFFFF",
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    # Convert canvas result to grayscale and resize to 28x28
    image = Image.fromarray(canvas_result.image_data.astype("uint8"), "RGBA")
    image = image.convert("L")
    image = image.resize((28, 28))

    # Normalize the image
    digit_image = 1 - np.array(image) / 255.0
    st.image(image, caption="Resized Image (28x28)", width=150)

    # Convert image to a list of lists
    digit_image = digit_image.tolist()

    # Send request
    if st.button("Predict"):
        response = requests.post("http://api:8000/predict/", json={"image": digit_image})
        prediction = response.json()["prediction"]
        st.write(f"Prediction: {prediction}")