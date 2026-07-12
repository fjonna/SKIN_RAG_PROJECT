import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000/diagnose-multimodal"

st.set_page_config(page_title="Skin Disease Multimodal Assistant", layout="centered")
st.title("Skin Disease Multimodal Assistant")
st.caption("Provide symptoms, a skin image, or both for informational analysis.")

symptoms = st.text_area("Describe skin symptoms (optional)")
uploaded_file = st.file_uploader(
    "Upload a skin image (optional)", type=["jpg", "jpeg", "png"]
)
top_k = st.slider("Number of candidates", min_value=1, max_value=5, value=3)


def analysis_mode(has_text: bool, has_image: bool) -> str:
    if has_text and has_image:
        return "Multimodal"
    if has_text:
        return "Text only"
    return "Image only"


if st.button("Analyze"):
    has_text = bool(symptoms.strip())
    has_image = uploaded_file is not None

    if not has_text and not has_image:
        st.warning("Please provide symptoms, an image, or both.")
    else:
        form_data = {"symptoms": symptoms, "top_k": top_k}
        # This harmless field ensures text-only requests still use multipart/form-data.
        files = {"_multipart": (None, "")}
        if has_image:
            files["file"] = (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type,
            )

        try:
            response = requests.post(API_URL, data=form_data, files=files, timeout=60)
            result = response.json()

            if response.status_code != 200:
                st.error(f"Backend error: {response.text}")
            elif "error" in result:
                st.error(result["error"])
            else:
                st.subheader(f"Analysis mode: {analysis_mode(has_text, has_image)}")
                for index, item in enumerate(result["candidates"], start=1):
                    st.markdown(f"### {index}. {item['name']}")
                    st.write(f"**Final score:** {item['final_score']}")
                    st.write(f"**Text score:** {item['text_score']}")
                    st.write(f"**Image score:** {item['image_score']}")
                    st.write(f"**Source:** {item['source']}")
                    st.write(f"**Description:** {item['description']}")
                    st.write(f"**Risk level:** {item['risk_level']}")
                    st.write(f"**Next steps:** {item['next_steps']}")
                    st.divider()

                st.info(result["disclaimer"])
        except requests.RequestException as error:
            st.error(f"Connection error: {error}")
