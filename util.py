import base64
from typing import Dict

import streamlit as st

from models.joint import runtime as joint_runtime
from models.stemmer import runtime as stemmer_runtime
from models.ner import runtime as ner_runtime


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_model(model_name: str, model_info: Dict):
    """Load a model."""
    model = None
    if model_name in {"JointModel-All", "JointModel-DependencyParsing", "JointModel-MorphemeTagging", "JointModel-MorphemeSegmentation"}:
        model = joint_runtime.load_model(
            model_info["model_path"], model_info["model_opts_path"]
        )
    elif model_name == "Stemmer":
        model = stemmer_runtime.load_model(model_info["model_path"])
    elif model_name == "NER":
        model = ner_runtime.load_model(model_info["model_path"], model_info["model_opts_path"])
    else:
        raise ValueError(f"Unknown model {type}")
    return model


class Doc:
    def __init__(self):
        self.dep = None
        self.morph = None
        self.morph_tag = None
        self.stemmed = None
        self.ner = None


# TODO: enable caching
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def process_text(model_name: str, model_info: Dict, text: str):
    """Process a text and create a Doc object."""
    model = load_model(model_name, model_info)
    doc = Doc()
    if model_name == "JointModel-All":
        doc.dep = joint_runtime.predict_dependecy(model, text)
        doc.morph = joint_runtime.predict_morphemes(model, text)
        doc.morph_tag = joint_runtime.predict_morpheme_tags(model, text)
    elif model_name == "JointModel-DependencyParsing":
        doc.dep = joint_runtime.predict_dependecy(model, text)
    elif model_name == "JointModel-MorphemeTagging":
        doc.morph_tag = joint_runtime.predict_morpheme_tags(model, text)
    elif model_name == "JointModel-MorphemeSegmentation":
        doc.morph = joint_runtime.predict_morphemes(model, text)
    elif model_name == "Stemmer":
        doc.stemmed = stemmer_runtime.predict_stems(model, text)
    elif model_name == "NER":
        doc.ner = ner_runtime.predict_ner(model, text)
    else:
        raise ValueError(f"Unknown model {type}")
    return doc


def get_svg(svg: str, style: str = "", wrap: bool = True):
    """Convert an SVG to a base64-encoded image."""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = f'<img src="data:image/svg+xml;base64,{b64}" style="{style}"/>'
    return get_html(html) if wrap else html


def get_img(img: str, style: str = "", wrap: bool = True):
    """Convert an SVG to a base64-encoded image."""
    b64 = base64.b64encode(open(img, "rb").read()).decode("utf-8")
    html = f'<img src="data:image/png;base64,{b64}" style="{style}"/>'
    return get_html(html) if wrap else html


def get_html(html: str):
    """Convert HTML so it can be rendered."""
    WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""
    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")
    return WRAPPER.format(html)


LOGO_IMG = "resources/logo_alt.png"
LOGO = get_img(LOGO_IMG, wrap=False, style="max-width: 100%; margin-bottom: 25px")
