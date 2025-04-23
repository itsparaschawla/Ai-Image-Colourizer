import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
from io import BytesIO
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# === Preprocessing Function ===
def preprocess_image(uploaded_image):
    img_rgb = np.array(uploaded_image.convert("RGB"))
    img_rgb = cv2.resize(img_rgb, (128, 128))
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2Lab)

    L_channel = img_lab[:, :, 0:1].astype(np.float32) / 255.0
    ab_channels = (img_lab[:, :, 1:].astype(np.float32) - 128.0) / 127.0
    return L_channel, ab_channels, img_rgb  # Return original resized RGB too

# === Lab to RGB Conversion ===
def lab_to_rgb(L, ab):
    L = L * 100.0
    ab = ab * 128.0
    lab = np.concatenate((L, ab), axis=-1).astype(np.float32)
    rgb = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    rgb = np.clip(rgb, 0, 1)
    return rgb

# === Custom Metrics for Loading Model ===
@tf.keras.utils.register_keras_serializable()
def ssim_metric(y_true, y_pred):
    y_true_scaled = (y_true + 1) / 2
    y_pred_scaled = (y_pred + 1) / 2
    return tf.reduce_mean(tf.image.ssim(y_true_scaled, y_pred_scaled, max_val=1.0))

@tf.keras.utils.register_keras_serializable()
def psnr_metric(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=2.0))

# === Load Model ===
@st.cache_resource
def load_colorization_model():
    return tf.keras.models.load_model("mountains_forest_u_net_best.h5", custom_objects={
        'ssim_metric': ssim_metric,
        'psnr_metric': psnr_metric,
        'mse': tf.keras.losses.MeanSquaredError()  # <- This is the fix
    })

model = load_colorization_model()

# === Streamlit UI ===
st.title("B/W Image Colorization 🌄🌲(Mountains & Forest)")
st.markdown("Upload a grayscale image of a mountain or forest landscape and get the colorized version with PSNR/SSIM metrics.")

uploaded_file = st.sidebar.file_uploader("Upload a black & white image of a mountain or forest landscape", type=["jpg", "png", "jpeg"])

# if uploaded_file:
#     uploaded_image = Image.open(uploaded_file)
#     st.image(uploaded_image, caption="Original Image", use_container_width=True)

#     with st.spinner("Colorizing..."):
#         L_channel, ab_true, original_resized = preprocess_image(uploaded_image)
#         input_L = np.expand_dims(L_channel, axis=0)  # (1, 128, 128, 1)

#         predicted_ab = model.predict(input_L)[0]  # (128, 128, 2)
#         colorized_image = lab_to_rgb(L_channel, predicted_ab)  # [0, 1] float

#         # Convert colorized image to uint8
#         colorized_uint8 = (colorized_image * 255).astype(np.uint8)

#         # Show colorized image
#         st.image(colorized_uint8, caption="Colorized Output", use_container_width=True)

if uploaded_file:
    uploaded_image = Image.open(uploaded_file)

    with st.spinner("Colorizing..."):
        L_channel, ab_true, original_resized = preprocess_image(uploaded_image)
        input_L = np.expand_dims(L_channel, axis=0)  # (1, 128, 128, 1)

        predicted_ab = model.predict(input_L)[0]  # (128, 128, 2)
        colorized_image = lab_to_rgb(L_channel, predicted_ab)  # [0, 1] float

        # Convert colorized image to uint8
        colorized_uint8 = (colorized_image * 255).astype(np.uint8)

    # Display side-by-side images
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_image, caption="Original Image", use_container_width=True)

    with col2:
        st.image(colorized_uint8, caption="Colorized Output", use_container_width=True)


        # Compute metrics using original resized image
        true_rgb = original_resized.astype(np.float32) / 255.0
        ssim_value = ssim(true_rgb, colorized_image, channel_axis=-1, data_range=1.0)
        psnr_value = psnr(true_rgb, colorized_image, data_range=1.0)

        st.markdown(f"**SSIM**: `{ssim_value:.4f}`  |  **PSNR**: `{psnr_value:.2f} dB`")

        # Download Button
        result_pil = Image.fromarray(colorized_uint8)
        buffer = BytesIO()
        result_pil.save(buffer, format="PNG")
        st.download_button(
            label="📥 Download Colorized Image",
            data=buffer.getvalue(),
            file_name="colorized_output.png",
            mime="image/png"
        )
