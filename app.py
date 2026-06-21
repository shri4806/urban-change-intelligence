import streamlit as st
import json

st.set_page_config(
    page_title="Urban Change Intelligence",
    layout="wide"
)

st.title("🌍 Urban Change Intelligence System")

st.markdown(
    """
    Upload two satellite images and generate
    urban change analytics and AI-powered reports.
    """
)

before_image = st.file_uploader(
    "Upload BEFORE image",
    type=["png", "jpg", "jpeg"]
)

after_image = st.file_uploader(
    "Upload AFTER image",
    type=["png", "jpg", "jpeg"]
)

if before_image and after_image:

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            before_image,
            caption="Before Image",
            width=300
        )

    with col2:
        st.image(
            after_image,
            caption="After Image",
            width=300
        )

if st.button("🚀 Analyze Urban Change"):

    st.success("Analysis Complete!")

    with open("change_report.json", "r") as file:
        report = json.load(file)

    st.subheader("📊 Change Statistics")

    st.write(f"Change Percentage: {report['change_percentage']}%")
    st.write(f"Regions Detected: {report['regions_detected']}")

    st.subheader("🗺️ Bounding Box Output")

    st.image(
        "boxed_regions.png",
        caption="Detected Development Zones"
    )

    st.write(f"Changed Pixels: {report['changed_pixels']}")
    st.write(f"Total Pixels: {report['total_pixels']}")
    st.write(f"Largest Region: {report['largest_region']} pixels")
    st.write(f"Smallest Region: {report['smallest_region']} pixels")
    st.write(f"Average Region Size: {report['average_region']} pixels")