import streamlit as st
import requests
from sklearn.cluster import KMeans


def sidebar(img, num_cluster):
    st.header("Parameters")
    num_cluster = st.slider("Number of clusters", 2, 25)

    image_option = st.radio(
        "Upload Image",
        (
            "URL",
            "Upload",
        ),
    )
    if image_option == "URL":
        img_data = st.text_input("URL for the image")
        if img_data != "":
            try:
                img = requests.get(img_data).content
                if img:
                    st.text("Uploaded image")
                    st.image(img)
            except:
                st.error("Invalid URL, please provide a URL that points to an image")
        if image_option == "Upload":
            data = st.file_uploader(
                "Please upload a valid image", type=["jpeg", "jpg", "png"]
            )
            if data:
                img = data.getvalue()
            try:
                if img:
                    st.text("Uploaded image")
                    st.image(img)
            except:
                st.error("Invalid Image")


if __name__ == "__main__":
    img = None
    num_cluster = 2
    with st.sidebar:
        sidebar(img, num_cluster)

    st.header("Image Segmentation Using KMeans")
    st.text("TODO: ADD project description/ how to use")
