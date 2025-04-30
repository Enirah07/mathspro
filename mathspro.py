import streamlit as st
import cv2
import numpy as np
from PIL import Image

def bilinear_resize(image_array, new_width, new_height):
    height, width, channels = image_array.shape
    resized_image = np.zeros((new_height, new_width, channels), dtype=np.uint8)

    x_ratio = float(width - 1) / (new_width - 1) if new_width > 1 else 0
    y_ratio = float(height - 1) / (new_height - 1) if new_height > 1 else 0

    for i in range(new_height):
        for j in range(new_width):
            x_l = int(x_ratio * j)
            y_t = int(y_ratio * i)
            x_h = min(x_l + 1, width - 1)
            y_b = min(y_t + 1, height - 1)

            x_weight = (x_ratio * j) - x_l
            y_weight = (y_ratio * i) - y_t

            a = image_array[y_t, x_l]
            b = image_array[y_t, x_h]
            c = image_array[y_b, x_l]
            d = image_array[y_b, x_h]

            pixel = (a * (1 - x_weight) * (1 - y_weight) +
                     b * x_weight * (1 - y_weight) +
                     c * y_weight * (1 - x_weight) +
                     d * x_weight * y_weight)

            resized_image[i, j] = pixel.astype(np.uint8)

    return resized_image

st.title("🔍 Bilinear Image Upscaler")
st.write("Upload an image to upscale it using bilinear interpolation")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    st.image(image, caption="Original Image", use_column_width=True)

    scale = st.slider("Upscale Factor", min_value=1, max_value=5, value=2)

    if st.button("Upscale"):
        new_width = image_array.shape[1] * scale
        new_height = image_array.shape[0] * scale
        result = bilinear_resize(image_array, new_width, new_height)
        st.image(result, caption="Upscaled Image", use_column_width=True)
