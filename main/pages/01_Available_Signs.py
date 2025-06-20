import streamlit as st
import os
from PIL import Image

st.set_page_config(layout="wide")
st.title("Available Hand Signs")

st.markdown("---")

# Get the directory of the current script to build a reliable path
script_dir = os.path.dirname(os.path.realpath(__file__))
# Path should go up two levels from 'main/pages' to the root 'ASL' directory
signs_folder = os.path.join(script_dir, '..', '..', 'signs')

try:
    sign_files = sorted([f for f in os.listdir(signs_folder) if f.lower().endswith('.png')])
    
    # Create columns for a grid layout
    num_columns = 3
    cols = st.columns(num_columns)
    
    for idx, fname in enumerate(sign_files):
        word = fname.replace('.png','').replace('_',' ')
        word = ' '.join(['I' if w.lower()=='i' else w.capitalize() for w in word.split()])
        
        image_path = os.path.join(signs_folder, fname)
        image = Image.open(image_path)

        with cols[idx % num_columns]:
            st.image(image, use_container_width=True)
            st.markdown(f"<h4 style='text-align: center;'>{word}</h4>", unsafe_allow_html=True)
            st.markdown("---")


except FileNotFoundError:
    st.error(f"Signs folder not found. Looked in: {signs_folder}") 