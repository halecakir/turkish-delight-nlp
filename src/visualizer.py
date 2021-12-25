from typing import Dict, List, Optional, Union

import pandas as pd
import streamlit as st
from spacy import displacy

from util import LOGO, get_html, get_svg, load_model, process_text

AVAILABLE_VISUALIZERS = {"parser", "morpheme_segmentation", "morpheme_tagging"}
# fmt: off
NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]
TOKEN_ATTRS = ["idx", "text", "lemma_", "pos_", "tag_", "dep_", "head", "morph",
               "ent_type_", "ent_iob_", "shape_", "is_alpha", "is_ascii",
               "is_digit", "is_punct", "like_num", "is_sent_start"]
# fmt: on
FOOTER = """<span style="font-size: 0.75em">&hearts; Built with [`spacy-streamlit`](https://github.com/explosion/spacy-streamlit)</span>"""


def visualize(
    models: Union[List[str], Dict[str, Dict]],
    default_text: str = "",
    default_model: Optional[str] = None,
    ner_labels: Optional[List[str]] = None,
    ner_attrs: List[str] = NER_ATTRS,
    token_attrs: List[str] = TOKEN_ATTRS,
    show_json_doc: bool = True,
    show_meta: bool = True,
    show_config: bool = True,
    show_visualizer_select: bool = False,
    show_model_info: bool = False,
    sidebar_title: Optional[str] = None,
    sidebar_description: Optional[str] = None,
    show_logo: bool = True,
    color: Optional[str] = "#b2dfdb",
    key: Optional[str] = None,
) -> None:
    """Embed the full visualizer with selected components."""

    if st.config.get_option("theme.primaryColor") != color:
        st.config.set_option("theme.primaryColor", color)

        # Necessary to apply theming
        st.experimental_rerun()

    if show_logo:
        st.sidebar.markdown(LOGO, unsafe_allow_html=True)
    if sidebar_title:
        st.sidebar.title(sidebar_title)
    if sidebar_description:
        st.sidebar.markdown(sidebar_description)

    # Allow both dict of model name / description as well as list of names
    model_names = models
    format_func = str
    if isinstance(models, dict):
        # format_func = lambda name: models.get(name, name)
        model_names = [name for name in models]

    default_model_index = (
        model_names.index(default_model)
        if default_model is not None and default_model in model_names
        else 0
    )
    selected_model = st.sidebar.selectbox(
        "Model",
        model_names,
        index=default_model_index,
        key=f"{key}_visualize_models",
        format_func=format_func,
    )
    model_load_state = st.info(f"Loading model '{selected_model}'...")
    load_model(selected_model, models[selected_model])
    model_load_state.empty()

    if show_model_info:
        st.sidebar.subheader("Model info")
        # TODO burayi duzenle
        desc = f"""<p style="font-size: 0.85em; line-height: 1.5">
                <strong>{selected_model}:</strong><br>
                <code>{models[selected_model]["code"]}</code>.<br>
                {models[selected_model]["description"]}</p>"""
        st.sidebar.markdown(desc, unsafe_allow_html=True)

    if show_visualizer_select:
        active_visualizers = st.sidebar.multiselect(
            "Visualizers",
            options=models[selected_model]["visualizers"],
            default=list(models[selected_model]["visualizers"]),
            key=f"{key}_viz_select",
        )

    text = st.text_area("Text to analyze", default_text, key=f"{key}_visualize_text")
    doc = process_text(selected_model, models[selected_model], text)
    if "parser" in AVAILABLE_VISUALIZERS and "parser" in active_visualizers:
        visualize_parser(doc.dep, key=key)
    if (
        "morpheme_segmentation" in AVAILABLE_VISUALIZERS
        and "morpheme_segmentation" in active_visualizers
    ):
        visualize_df(doc.morph)
    if (
        "morpheme_tagging" in AVAILABLE_VISUALIZERS
        and "morpheme_tagging" in active_visualizers
    ):
        visualize_df(
            doc.morph_tag, title="Morpheme Tagging", colorized_col="morpheme_tags"
        )


def visualize_parser(
    doc,
    *,
    title: Optional[str] = "Dependency Parse & Part-of-speech tags",
    key: Optional[str] = None,
) -> None:
    """Visualizer for dependency parses."""
    if title:
        st.header(title)
    cols = st.columns(4)

    options = {
        "compact": cols[3].checkbox("Compact mode", key=f"{key}_parser_compact"),
    }
    html = displacy.render(doc, options=options, style="dep", manual=True)
    html = html.replace("\n\n", "\n")
    st.write(get_svg(html), unsafe_allow_html=True)


def visualize_df(
    doc,
    *,
    title: Optional[str] = "Morpheme Segmentation",
    colorized_col: Optional[str] = "morphemes",
) -> None:
    """Visualizer for text categories."""
    if title:
        st.header(title)
    df = pd.DataFrame(doc)
    df = df.assign(order=range(len(df))).set_index("order")
    style = df.style.set_properties(
        **{"background-color": "#ffffb3"}, subset=[colorized_col]
    )
    st.table(style)
