from loader import DataLoader
from utils import load_data
import yaml
from ydata_profiling import ProfileReport
import streamlit as st
import streamlit.components.v1 as components
import seaborn as sns
import pandas as pd
from plots import mutual_info_to_target, plot_pca_cumulative_variance, plot_pca


with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)


def eda(session: st):

    state = session.session_state
    for var in [
        'data_loader', 'data_profile_html',
        'pairplot_fig', 'mutual_info_fig',
        'pca_fig', 'pca_var_fig'
    ]:
        if var not in state:
            state[var] = None

    session.markdown('## Choose The Parameters')
    col1, col2 = session.columns(2)
    with col1:
        threshold = session.text_input('Threshold for Classes:', key='data_threshold')
    with col2:
        embedding_size = session.number_input('Size of Embeddings:', key='data_embedding_size')

    if session.button('Generate!', key='data_generate', use_container_width=True):

        with session.spinner("Loading Data..."):

            print('selected task:', state.selected_task)
            loader = load_data(
                loader_type=state.selected_method,
                task=state.selected_task,
                fingerprint_type=state.fingerprint_type,
                threshold=threshold,
                test_size=config['test_size'],
                calibration_size=config['calibration_size'],
                descriptors=None,
                embedding_size=embedding_size
            )
            state.data_loader = loader

        with session.spinner("Generating Profile..."):

            profiler = ProfileReport(loader.dataset, minimal=True)
            state.data_profile_html = profiler.to_html()

        state.pairplot_fig = None
        state.mutual_info_fig = None
        state.pca_fig = None
        state.pca_var_fig = None

    loader = state.data_loader

    if loader is not None:

        session.markdown('### Data Overview')
        session.dataframe(loader.dataset.head(10))

        session.markdown('### Data Profile')
        components.html(state.data_profile_html, height=700, scrolling=True)

        session.divider()

        session.markdown('## Pair Plots')

        with session.form('pairplot_form', clear_on_submit=False):
            subset = session.multiselect(
                'Select features',
                options=loader.dataset.columns.tolist(),
                key='data_subset'
            )
            kind = session.selectbox(
                'Kind',
                ['scatter', 'kde', 'hist', 'reg'],
                key='data_kind'
            )
            diag = session.selectbox(
                'Diag Kind',
                ['auto', 'hist', 'kde', None],
                key='data_diag_kind'
            )
            plot_pair = session.form_submit_button('Plot Pairplot', use_container_width=True)

        if plot_pair:
            if not subset:
                session.warning("Select at least one feature.")
            else:
                with session.spinner("Rendering Pair Plot..."):
                    fig = sns.pairplot(
                        loader.dataset[subset],
                        kind=kind,
                        diag_kind=diag,
                        markers='+'
                    ).figure
                    state.pairplot_fig = fig

        if state.pairplot_fig:
            session.pyplot(state.pairplot_fig)

        session.divider()

        session.markdown('## Mutual Information with Target')

        with session.form('mutual_form', clear_on_submit=False):
            target_mi = session.selectbox(
                'Target Variable',
                options=loader.dataset.columns.tolist(),
                index=loader.dataset.columns.get_loc('target'),
                key='data_target_var'
            )
            max_feat = session.number_input(
                'Max Features',
                min_value=1,
                max_value=len(loader.dataset.columns),
                value=10,
                key='data_maximum_features'
            )
            plot_mi = session.form_submit_button('Calculate MI', use_container_width=True)

        if plot_mi:
            with session.spinner("Calculating Mutual Information..."):
                try:
                    fig = mutual_info_to_target(
                        df=loader.dataset,
                        target=target_mi,
                        class_problem='binary' if state.selected_task == 'classification' else 'regression',
                        maximum_features=max_feat
                    ).figure
                    state.mutual_info_fig = fig
                except ValueError:
                    state.mutual_info_fig = None
                    session.warning("This is not supported for classification tasks.")

        if state.mutual_info_fig:
            session.pyplot(state.mutual_info_fig)

        session.divider()

        session.markdown('## Principal Component Analysis (PCA)')

        with session.form('pca_form', clear_on_submit=False):

            target_pca = session.selectbox(
                'Target Variable for PCA',
                options=loader.dataset.columns.tolist(),
                index=loader.dataset.columns.get_loc('target'),
                key='data_pca_target_var'
            )

            n_comp = session.number_input(
                'Number of Components',
                min_value=2,
                max_value=len(loader.dataset.columns),
                value=2,
                key='data_pca_components'
            )

            plot_pca_btn = session.form_submit_button('Plot PCA', use_container_width=True)

        if plot_pca_btn:
            with session.spinner("Performing PCA..."):
                
                fig = plot_pca(
                    df=loader.dataset,
                    target=target_pca,
                )
                state.pca_fig = fig

                fig_var = plot_pca_cumulative_variance(
                    loader.dataset.drop(columns=[target_pca]),
                    n_components=n_comp
                ).figure
                
                state.pca_var_fig = fig_var

        if state.pca_fig:
            session.pyplot(state.pca_fig)
        if state.pca_var_fig:
            session.pyplot(state.pca_var_fig)
