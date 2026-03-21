import streamlit as st
import requests

st.title("Skin Disease Assistant")
st.write("Enter your skin symptoms and get top candidate diseases.")

symptoms = st.text_area("Describe your symptoms")
top_k = st.slider("Number of candidates", 1, 5, 3)

if st.button("Analyze"):
    if symptoms.strip() == "":
        st.warning("Please enter your symptoms first.")
    else:
        response = requests.post(
            "http://127.0.0.1:8000/diagnose",
            json={"symptoms": symptoms, "top_k": top_k}
        )

        if response.status_code == 200:
            data = response.json()
            st.subheader("Top candidates")

            for i, item in enumerate(data["candidates"], start=1):
                st.markdown(f"### {i}. {item['name']}")
                st.write(f"**Confidence:** {item['confidence']}")
                st.write(f"**Description:** {item['description']}")
                st.write(f"**Risk level:** {item['risk_level']}")
                st.write(f"**Next steps:** {item['next_steps']}")
                st.divider()

            st.info(data["disclaimer"])
        else:
            st.error("Something went wrong while contacting the backend.")