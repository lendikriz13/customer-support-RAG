"""
Simple test to verify Streamlit is working
"""
import streamlit as st

st.set_page_config(page_title="Test App", page_icon="🧪")

st.title("🧪 Streamlit Test App")
st.write("If you can see this, Streamlit is working!")

st.success("✅ Success! The app is rendering correctly.")

if st.button("Click me!"):
    st.balloons()
    st.write("Button clicked!")
