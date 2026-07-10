import streamlit as st
import requests

st.set_page_config(page_title="Skin Disease Assistant", layout="centered")

st.title("Skin Disease Assistant")

tab1, tab2, tab3 = st.tabs(["Text Search", "Image Search", "Multimodal Analysis"])

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

with tab3:
    st.subheader("Analyze symptoms and/or a skin image")

    multimodal_symptoms = st.text_area(
        "Describe your symptoms (optional)", key="multimodal_symptoms"
    )
    multimodal_file = st.file_uploader(
        "Choose an image (optional)",
        type=["jpg", "jpeg", "png"],
        key="multimodal_uploader",
    )
    top_k_multimodal = st.slider(
        "Number of combined candidates", 1, 5, 3, key="multimodal_slider"
    )

    if st.button("Analyze Multimodal"):
        try:
            form_data = {
                "symptoms": multimodal_symptoms,
                "top_k": top_k_multimodal,
            }
            # Supplying a harmless form part keeps text-only requests multipart/form-data.
            files = {"_multipart": (None, "")}
            if multimodal_file is not None:
                files["file"] = (
                    multimodal_file.name,
                    multimodal_file.getvalue(),
                    multimodal_file.type,
                )

            response = requests.post(
                "http://127.0.0.1:8000/diagnose-multimodal",
                data=form_data,
                files=files,
            )
            result = response.json()

            if response.status_code != 200:
                st.error(f"Backend error: {response.text}")
            elif "error" in result:
                st.error(result["error"])
            else:
                st.subheader("Combined candidates")
                for i, item in enumerate(result["candidates"], start=1):
                    st.markdown(f"### {i}. {item['name']}")
                    st.write(f"**Final score:** {item['final_score']}")
                    st.write(f"**Text score:** {item['text_score']}")
                    st.write(f"**Image score:** {item['image_score']}")
                    st.write(f"**Source:** {item['source']}")
                    st.write(f"**Description:** {item['description']}")
                    st.write(f"**Risk level:** {item['risk_level']}")
                    st.write(f"**Next steps:** {item['next_steps']}")
                    st.divider()

                st.info(result["disclaimer"])
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
