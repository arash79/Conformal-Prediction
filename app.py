import streamlit as st
import pandas as pd
from PIL import Image
import base64
import yaml
from tabs import *


pd.set_option("display.precision", 2)


st.set_page_config(page_title="Analytical Dashboard", page_icon=":mag:", layout="wide")


st.markdown(
    """
    <style>
        /* Sidebar background */
        section[data-testid="stSidebar"] {
            background-color: #fad390;
        }
        /* Custom object styling */
        .custom-object {
            margin: 0px;
            padding: 0px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_config(config_file="config.yaml"):
    try:
        with open(config_file, "r", encoding="utf-8-sig") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error(f'{config_file} not found!')
        return {}


def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.error(f"{image_path} not found!") 
        return None


def display_header(logo_path, title):

    logo_base64 = get_base64_image(logo_path)

    if logo_base64:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center;">
                <img src="data:image/png;base64,{logo_base64}" alt="Logo" style="height: 200px; margin-right: 10px;">
                <h1 style="display: inline; margin: 0;">{title} 🔍</h1>
            </div>
            """,
            unsafe_allow_html=True,
        )

def display_sidebar():

    with st.sidebar:

        st.image(Image.open("images/logo.png"), use_column_width=True)

        st.divider()

        tasks = ['classification', 'regression']

        selected_task = st.selectbox(
            "Please Select One of the Following Tasks:",
            tasks,
            index=tasks.index(st.session_state.get("selected_task", "classification")))

        methods = ['descriptors', 'fingerprints', 'graph']

        selected_method = st.selectbox(
            "Please Select One of the Following Methods:",
            methods,
            index=methods.index(st.session_state.get("selected_method", "descriptors")))

        if selected_method == 'fingerprints':

            fingerprint_type = ['morgan', 'rdk', 'atom pair', 'topological torsion']

            selected_fingerprint = st.selectbox(
            "Please Select One of the Following Fingerprint Types:",
            fingerprint_type,
            index=fingerprint_type.index(st.session_state.get("selected_fingerprint", "morgan")))

        if st.button("Go!", use_container_width=True): 
            st.session_state.selected_task = selected_task
            st.session_state.selected_method = selected_method
            st.session_state.fingerprint_type = selected_fingerprint if selected_method == 'fingerprints' else None


def initialize_session_state():

    defaults = {
        "selected_task": "classification",
        "selected_method": "descriptors",
        "fingerprint_type": "morgan",
        "selected_technique": "Mutual Information",
    }

    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def main():

    initialize_session_state()

    display_header(logo_path='images/image.png', title="Analytical Dashboard")

    st.divider()

    display_sidebar()

    try:
        initialize_session_state()
    except Exception as e:
        pass

    tabs = [
        "EDA",
        "Classification",
        "Regression"
    ]

    tab_functions = [
        eda,
        classification,
        regression
    ]

    for tab, func in zip(st.tabs(tabs), tab_functions):
        with tab:
            func(session=st)


if __name__ == "__main__":
    main()
