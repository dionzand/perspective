import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates
import numpy as np
import cv2

st.set_page_config(
    page_title="Streamlit Image Coordinates: Image Update",
    page_icon="🎯",
    layout="wide",
)


if "current_image" not in st.session_state:
    with st.sidebar:
        uploaded_image = st.file_uploader("Open image")
        if "previous_image" in st.session_state:
            st.write("Previous image")
            st.image(st.session_state["previous_image"])

        if uploaded_image:
            current_image = Image.open(uploaded_image)
            original_height, original_width = current_image.size
            aspect_ratio = original_width / original_height
            new_height = int(1200 * aspect_ratio)

            resized_image = cv2.resize(np.array(current_image), (1200, new_height))
            st.session_state["current_image"] = resized_image
            st.experimental_rerun()

else:
    if "points" not in st.session_state:
        st.session_state["points"] = []

    if len(st.session_state["points"]) != 4:
        coordinates = streamlit_image_coordinates(st.session_state["current_image"], key="total")

        if coordinates is not None:
            point = coordinates["x"], coordinates["y"]

            if point not in st.session_state["points"]:
                st.session_state["points"].append(point)
                st.experimental_rerun()

    if len(st.session_state["points"]) == 4:
        src_pts = np.array(st.session_state["points"], dtype=np.float32)
        dst_pts = np.array([[0, 0], [500, 0], [500, 500], [0, 500]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warp = cv2.warpPerspective(st.session_state["current_image"], M, (500, 500))
        HSV_img = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)

        with st.sidebar:
            blur_value = st.number_input("Blur radius", min_value=0, max_value=100, value=0, step=1)
            contrast = st.number_input("Contrast", min_value=0.01, max_value=3.0, value=1.6, step=0.01)
            brightness = st.number_input("Brightness", min_value=-50, max_value=50, value=-2, step=1)
            if "previous_image" in st.session_state:
                st.write("Previous image")
                st.image(st.session_state["previous_image"])

        if blur_value > 0:
            HSV_img = cv2.blur(HSV_img, (blur_value, blur_value))

        HSV_img = cv2.convertScaleAbs(HSV_img, alpha=contrast, beta=brightness)
        st.image(HSV_img)

        if "previous_image" in st.session_state:
            image_difference = cv2.absdiff(st.session_state["previous_image"], HSV_img)

            with st.sidebar:
                mask_contrast = st.number_input("Mask contrast", min_value=0.01, max_value=3.0, value=1.6, step=0.01)
                mask_threshold = st.number_input("Mask threshold", min_value=0, max_value=255, value=20, step=1,
                                                 help="Minimal value (0-255) of pixel to be included in mask")


            st.write("Difference with previous image")
            image_difference = cv2.cvtColor(image_difference, cv2.COLOR_BGR2GRAY)

            image_difference = cv2.convertScaleAbs(image_difference, alpha=mask_contrast)
            st.image(image_difference)

            _, binary_mask = cv2.threshold(image_difference, mask_threshold, 255, cv2.THRESH_BINARY)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

            with st.sidebar:
                min_size = st.number_input("Min size", min_value=0, max_value=1000, value=20, step=1,
                                           help="Minimal size of pixel cluster to be included in mask")
            result_mask = np.zeros_like(image_difference, dtype=np.uint8)

            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= min_size:
                    result_mask[labels == i] = 255

            st.write("Mask difference image")
            st.image(result_mask)

        if st.button("Go to next image"):
            st.session_state["previous_image"] = HSV_img
            del st.session_state["current_image"]
            del st.session_state["points"]
            st.experimental_rerun()
