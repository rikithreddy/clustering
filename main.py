import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
from sklearn.cluster import KMeans


@st.cache
def _gen_mss(t_img):
    x, y, z = t_img.shape
    t_img = t_img.reshape(x * y, z)
    cluster_mss = []
    k_values = range(1, 10)
    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans = kmeans.fit(t_img)
        cluster_mss.append(kmeans.inertia_)
    return cluster_mss


def _im_to_np(img):
    im = Image.open(io.BytesIO(img))
    im = np.asarray(im)
    return im


def _handle_url():
    img = None
    img_data = st.text_input(
        "URL for the image",
        value="https://lh6.googleusercontent.com/TXpJmvrCYHRyK6gSTjYbPmYDYhu4m8wNHg22nyKYiJvMJ55hXLkY-xyNDyaLcbwh_CrjYtdsMoAAtpARdibxolo=w1280",
    )
    if img_data == "":
        return
    try:
        img = requests.get(img_data).content
        if img:
            st.text("Uploaded image")
            st.image(img)
    except:
        st.error("Invalid URL, please provide a URL that points to an image")
    return img


def _handle_upload():
    img = None
    data = st.file_uploader("Please upload a valid image", type=["jpeg", "jpg", "png"])
    if data:
        img = data.getvalue()
    try:
        if img:
            st.text("Uploaded image")
            st.image(img)
    except:
        st.error("Invalid Image")
    return img


def handle_image(image_option):
    img = _handle_url() if image_option == "URL" else _handle_upload()
    return img


def handle_sidebar():

    # Added only for demo!
    img, num_cluster = None, 2

    st.header("Parameters")
    # Create a slider input
    num_cluster = st.slider("Number of clusters", 2, 25)
    # Create Radio input
    image_option = st.radio(
        "Upload Image",
        (
            "URL",
            "Upload",
        ),
    )
    img = handle_image(image_option)
    return img, num_cluster


@st.cache(suppress_st_warning=True)
def segment_img(im, num_cluster):
    x, y, z = im.shape
    img = im.reshape(x * y, z)
    model = KMeans(n_clusters=num_cluster)
    model.fit(img)
    new_clusters = []
    for clabel in model.labels_:
        new_clusters.append(model.cluster_centers_[clabel])

    seg = np.array(new_clusters).reshape(x, y, z).astype("int")

    PIL_image = Image.fromarray(seg.astype("uint8"), "RGB")
    return PIL_image


def handle_body(img, num_cluster):
    if not img:
        return

    original, segmented = st.columns(2)

    original.header("Original")
    original.image(img)

    t_img = _im_to_np(img)

    seg_img = segment_img(t_img, num_cluster)

    segmented.header("Segmented Image")
    segmented.image(seg_img)

    # Want to find best clusters?
    elbow = st.checkbox("Show Elbow curve? (This can be time consuming)")

    if elbow:
        cluster_mss = _gen_mss(t_img)
        st.line_chart(cluster_mss)


if __name__ == "__main__":
    img, num_cluster = None, 2
    # Add a header, and some text
    st.header("Image Segmentation Using KMeans")
    st.text("Please add a URL on the original and select number of colors you want")

    # Let's make a sidebar
    with st.sidebar:
        img, num_cluster = handle_sidebar()

    handle_body(img, num_cluster)
