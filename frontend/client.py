import streamlit as st
import requests

st.title("Skin Disease Assistant")
st.write("Enter your skin symptoms and get an informational suggestion.")

symptoms = st.text_area("Describe your symptoms")

if st.button("Analyze"):
    if symptoms.strip() == "":
        st.warning("Please enter your symptoms first.")
    else:
        response = requests.post(
            "http://127.0.0.1:8000/diagnose",
            json={"symptoms": symptoms}
        )

        if response.status_code == 200:
            data = response.json()
            st.subheader("Result")
            st.write("**Most likely disease:**", data["most_likely_disease"])
            st.write("**Description:**", data["description"])
            st.write("**Risk level:**", data["risk_level"])
            st.write("**Next steps:**", data["next_steps"])
            st.write("**Similarity score:**", data["similarity_score"])
            st.info(data["disclaimer"])
        else:
            st.error("Something went wrong while contacting the backend.")