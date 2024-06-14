import streamlit as st

st.title("Redirect to the new link for EER Chat")

# Function to generate JavaScript for redirecting
def js_redirect(url):
    st.write(f'<meta http-equiv="refresh" content="0; URL={url}">', unsafe_allow_html=True)

# Button to trigger redirect
if st.button("Go to EERChat"):
    js_redirect("https://eerchat.ploomberapp.io/")