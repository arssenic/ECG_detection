import streamlit as st
import numpy as np
import cv2
from PIL import Image as PILImage
from io import BytesIO
from streamlit_cropperjs import st_cropperjs
import base64
import matplotlib.pyplot as plt
import pandas as pd
from signal_detection import ColorImage, adaptive
from signal_extract import extractSignal, BinaryImage

st.set_page_config(page_title="ECG Signal Processor", layout="wide")
st.title("🩺 ECG Signal Detection and Extraction Tool")

uploaded_file = st.file_uploader("Upload ECG Scan Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original = cv2.imdecode(file_bytes, 1)
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    final_image = original_rgb.copy()

    option = st.selectbox("Select Action", ["Rotate", "Crop", "Rotate & Crop"])

    if option == "Rotate":
        angle = st.slider("Rotate Image (degrees)", -180, 180, 0)
        if angle != 0:
            center = (final_image.shape[1] // 2, final_image.shape[0] // 2)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            final_image = cv2.warpAffine(
                final_image,
                rot_mat,
                (final_image.shape[1], final_image.shape[0]),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
        st.subheader("🔄 Rotated Image Preview")
        st.image(final_image, use_container_width=True)

    elif option == "Crop":
        st.subheader("✂️ Draw rectangle to crop")
        original_pil = PILImage.fromarray(final_image)
        buffer = BytesIO()
        original_pil.save(buffer, format="PNG")
        buffer.seek(0)
        cropped_pic = st_cropperjs(pic=buffer.read(), btn_text="Crop!", key="crop_only")

        if cropped_pic:
            image_data = base64.b64decode(cropped_pic.split(",")[1])
            final_image = PILImage.open(BytesIO(image_data)).convert("RGB")
            final_image = np.array(final_image)
            st.image(final_image, output_format="PNG")

    elif option == "Rotate & Crop":
        st.subheader("Step 1️⃣: Rotate")
        angle = st.slider("Rotate Image (degrees)", -180, 180, 0, key="angle_slider")
        center = (original_rgb.shape[1] // 2, original_rgb.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            original_rgb,
            rot_mat,
            (original_rgb.shape[1], original_rgb.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        st.subheader("🔄 Rotated Image Preview")
        st.image(rotated, use_container_width=True)

        rotated_pil = PILImage.fromarray(rotated)
        buffer = BytesIO()
        rotated_pil.save(buffer, format="PNG")
        buffer.seek(0)

        st.subheader("Step 2️⃣: Crop Rotated Image")
        cropped_pic = st_cropperjs(pic=buffer.read(), btn_text="Crop!", key=f"rotate_crop_{angle}")

        if cropped_pic:
            image_data = base64.b64decode(cropped_pic.split(",")[1])
            final_image = PILImage.open(BytesIO(image_data)).convert("RGB")
            final_image = np.array(final_image)
            st.image(final_image, output_format="PNG")
        else:
            final_image = rotated

    # Allow user to download the processed (rotated/cropped) image
    buffered = BytesIO()
    result_pil = PILImage.fromarray(final_image) if isinstance(final_image, np.ndarray) else final_image
    result_pil.save(buffered, format="PNG")
    buffered.seek(0)
    st.download_button(
        label="📥 Download Processed Image",
        data=buffered,
        file_name="processed_ecg.png",
        mime="image/png"
    )

    # Signal Detection Section
    st.subheader("📈 Run Signal Detection and Extraction")
    if st.button("Detect & Extract Signal"):
        with st.spinner("Processing..."):
            # --- Signal Detection ---
            color_img = ColorImage(cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
            binary_output = adaptive(color_img)
            signal_mask = (binary_output.data * 255)
            st.image(signal_mask, caption="Detected Signal Mask", use_container_width=True)

            # Download Signal Mask
            signal_pil = PILImage.fromarray(signal_mask)
            sig_buffer = BytesIO()
            signal_pil.save(sig_buffer, format="PNG")
            sig_buffer.seek(0)
            st.download_button(
                label="📥 Download Detected Signal Mask",
                data=sig_buffer,
                file_name="detected_signal.png",
                mime="image/png"
            )

            # --- Signal Extraction ---
            _, binary = cv2.threshold(signal_mask, 127, 1, cv2.THRESH_BINARY)

            # Wrap it into a BinaryImage
            binary_image = BinaryImage(binary)

            # Now call extractSignal
            signal = extractSignal(binary_image)

            # Extract x and y values if signal is not None
            if signal is not None:
                x = np.arange(len(signal))
                y = -signal  # Flip vertically if needed

                # Plotting the signal
                fig, ax = plt.subplots(figsize=(20, 2.5))
                ax.plot(x, y, color="blue")
                ax.set_title("Extracted ECG Signal")
                st.pyplot(fig)

                # Download CSV
                csv = pd.DataFrame({'Time': x, 'Amplitude': y}).to_csv(index=False).encode('utf-8')
                st.download_button("📥 Download Signal as CSV", csv, "ecg_signal.csv", "text/csv")

                # Download Plot Image
                plot_buf = BytesIO()
                fig.savefig(plot_buf, format="png")
                plot_buf.seek(0)
                st.download_button("📥 Download Plot Image", plot_buf.getvalue(), "ecg_plot.png", "image/png")

                st.subheader("🔍 Comparison: Cropped Image vs Extracted Signal")

                # Resize both images to the same size (e.g., 600x200)
                resize_width = 600
                resize_height = 200

                # Convert and resize cropped image
                cropped_img_pil = PILImage.fromarray(final_image).resize((resize_width, resize_height))

                # Generate extracted signal plot with border and limited y-axis scaling
                fig2, ax2 = plt.subplots(figsize=(6, 2))
                ax2.plot(x, y, color="purple", linewidth=2)
                ax2.set_title("Extracted Signal", fontsize=12, weight='bold')
                ax2.set_facecolor("white")

                # Add visible border
                for spine in ax2.spines.values():
                    spine.set_edgecolor('black')
                    spine.set_linewidth(1.5) 

                signal_plot_buf = BytesIO()
                fig2.savefig(signal_plot_buf, format="png", bbox_inches="tight")
                plt.close(fig2)
                signal_plot_buf.seek(0)
                signal_plot_image = PILImage.open(signal_plot_buf).resize((resize_width, resize_height))

                # Side-by-side display with fixed width and height
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cropped_img_pil, caption="🖼️ Cropped ECG Image", use_container_width=False)
                with col2:
                    st.image(signal_plot_image, caption="📈 Extracted Signal Plot", use_container_width=False)
