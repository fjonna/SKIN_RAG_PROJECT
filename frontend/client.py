import streamlit as st
import requests

st.set_page_config(page_title="Skin Disease Assistant", layout="centered")

st.title("Skin Disease Assistant")

tab1, tab2 = st.tabs(["Text Search", "Image Search"])

with tab1:
    st.subheader("Describe symptoms")
    symptoms = st.text_area("Describe your symptoms")

    top_k_text = st.slider(
        "Number of text candidates",
        1,
        5,
        3,
        key="text_slider"
    )

    if st.button("Analyze Text"):
        if symptoms.strip() == "":
            st.warning("Please enter your symptoms first.")
        else:
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/diagnose",
                    json={"symptoms": symptoms, "top_k": top_k_text}
                )

                if response.status_code == 200:
                    data = response.json()
                    st.subheader("Top text candidates")

                    for i, item in enumerate(data["candidates"], start=1):
                        st.markdown(f"### {i}. {item['name']}")
                        st.write(f"**Confidence:** {item['confidence']}")
                        st.write(f"**Description:** {item['description']}")
                        st.write(f"**Risk level:** {item['risk_level']}")
                        st.write(f"**Next steps:** {item['next_steps']}")
                        st.divider()

                    st.info(data["disclaimer"])
                else:
                    st.error(f"Backend error: {response.text}")

            except Exception as e:
                st.error(f"Connection error: {e}")

with tab2:
    st.subheader("Upload skin image")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    top_k_img = st.slider(
        "Number of image candidates",
        1,
        5,
        3,
        key="img_slider"
    )

    if st.button("Analyze Image"):
        if uploaded_file is None:
            st.warning("Please upload an image first.")
        else:
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type
                    )
                }

                data = {"top_k": top_k_img}

                response = requests.post(
                    "http://127.0.0.1:8000/diagnose-image",
                    files=files,
                    data=data
                )

                if response.status_code == 200:
                    result = response.json()
                    st.subheader("Top image candidates")

                    for i, item in enumerate(result["candidates"], start=1):
                        st.markdown(f"### {i}. {item['label']}")
                        st.write(f"**Confidence:** {item['confidence']}")
                        st.write(f"**Matches in top results:** {item['matches']}")
                        st.caption(item["example_path"])
                        st.divider()

                    st.info(result["disclaimer"])
                else:
                    st.error(f"Backend error: {response.text}")

            except Exception as e:
                st.error(f"Connection error: {e}")