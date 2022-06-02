import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
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
    return img, num_cluster


if __name__ == "__main__":
    img = None
    num_cluster = 2
    with st.sidebar:
        img, num_cluster = sidebar(img, num_cluster)

    st.header("Image Segmentation Using KMeans")
    st.text("TODO: ADD project description/ how to use")
    if img:
        left, right = st.columns(2)
        left.header("Original")
        left.image(img)

        im = np.asarray(Image.open(io.BytesIO(img)))
        x, y, z = im.shape
        t_img = im.reshape(x * y, z)

        model = KMeans(n_clusters=num_cluster)
        model.fit(t_img)

        new_clusters = []
        for clabel in model.labels_:
            new_clusters.append(model.cluster_centers_[clabel])

        seg = np.array(new_clusters).reshape(x, y, z).astype("int")
        PIL_image = Image.fromarray(seg.astype("uint8"), "RGB")

        # seg_img = cv2.imencode(".png", seg)
        right.header("Segmented colors")
        right.image(seg)

        elbow = st.checkbox("Show Elbow curve? (This can be time consuming)")

        if elbow:
            cluster_mss = []
            k_values = range(1, 20)
            for k in k_values:
                kmeans = KMeans(n_clusters=k)
                kmeans = kmeans.fit(t_img)
                cluster_mss.append(kmeans.inertia_)
            st.line_chart(cluster_mss)
