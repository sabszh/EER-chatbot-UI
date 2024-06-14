import streamlit as st

# Set the target URL
target_url = "https://eerchat.ploomberapp.io/"

# Redirect to the target URL
st.write(f"Redirecting to the new [EER Chat]({target_url}) link on Ploomber...")
st.markdown(f'<meta http-equiv="refresh" content="0; url={target_url}">', unsafe_allow_html=True)
