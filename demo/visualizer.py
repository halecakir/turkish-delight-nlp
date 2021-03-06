from typing import Dict, List, Optional, Union
from io import StringIO
import pandas as pd
import streamlit.report_thread as ReportThread
from threading import Thread
import streamlit as st
from spacy import displacy
import os
from util import LOGO, get_svg, process_text, get_html

ctx = ReportThread.get_report_ctx()
AVAILABLE_VISUALIZERS = {
    "parser",
    "morpheme_segmentation",
    "morpheme_tagging",
    "stemming",
    "ner",
}
# fmt: off
NER_ATTRS = ["text", "label_", "start", "end", "start_char", "end_char"]
TOKEN_ATTRS = ["idx", "text", "lemma_", "pos_", "tag_", "dep_", "head", "morph",
               "ent_type_", "ent_iob_", "shape_", "is_alpha", "is_ascii",
               "is_digit", "is_punct", "like_num", "is_sent_start"]
# fmt: on
FOOTER = """<span style="font-size: 0.75em">&hearts; Built with [`spacy-streamlit`](https://github.com/explosion/spacy-streamlit)</span>"""

from PIL import Image


def visualize(
    models: Union[List[str], Dict[str, Dict]],
    default_model: Optional[str] = None,
    show_model_info: bool = True,
    sidebar_title: Optional[str] = None,
    sidebar_description: Optional[str] = None,
    show_logo: bool = True,
    color: Optional[str] = "#b2dfdb",
    papers_md: str = "",
    contact_md: str = "",
    rest_url:str = ""
) -> None:
    """Embed the full visualizer with selected components."""

    def draw():
        selected_model = st.session_state.item
        if selected_model != "Select Model":
            text = st.text_area(
                "Please enter sentence",
                models[selected_model]["deafult_sentence"],
                on_change=draw,
            )
            st.info(
                "Note: Currently, we don't support sentence segmentation, the selected model works best with a single sentence!"
            )
            doc = process_text(selected_model, models[selected_model], text)
        if selected_model == "JointModel-DependencyParsing":
            visualize_parser(doc.dep)
        elif selected_model == "JointModel-MorphemeSegmentation":
            visualize_df(doc.morph)
        elif selected_model == "JointModel-MorphemeTagging":
            visualize_df(
                doc.morph_tag, title="Morpheme Tagging", colorized_col="morpheme_tags"
            )
        elif selected_model == "JointModel-PoSTagging":
            visualize_df(doc.pos, title="PoS Tagging", colorized_col="pos")
        elif selected_model == "JointModel-All":
            visualize_parser(doc.dep)
            visualize_df(doc.pos, title="PoS Tagging", colorized_col="pos")
            visualize_df(doc.morph)
            visualize_df(
                doc.morph_tag, title="Morpheme Tagging", colorized_col="morpheme_tags"
            )
        elif selected_model == "Stemmer":
            visualize_df(doc.stemmed, title="Stemming", colorized_col="stems")
        elif selected_model == "NER":
            st.warning(
                "Warning : please note that the NER model is trained on small amount of social media text. It could be further trained on larger or formal text for better performance."
            )
            visualize_ner(doc.ner)
        elif selected_model == "SemanticParser":
            image = Image.open(doc.ucca)
            st.image(image, caption="Semantic Parser Result")

    def show_contacts():
        with open(contact_md) as target:
            content = target.read()
            st.markdown(content)

    def others():
        with open(papers_md) as target:
            content = target.read()
            st.markdown(content)

    st.set_page_config(
        page_title=sidebar_title,
        page_icon=":koala:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if st.config.get_option("theme.primaryColor") != color:
        st.config.set_option("theme.primaryColor", color)

        # Necessary to apply theming
        st.experimental_rerun()

    st.markdown(
        """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 500px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 500px;
        margin-left: -500px;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )
    if show_logo:
        st.sidebar.markdown(LOGO, unsafe_allow_html=True)
    if sidebar_description:
        st.sidebar.markdown(sidebar_description)

    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

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
        # index=default_model_index,
        # format_func=format_func,
        on_change=draw,
        key="item",
    )
    model_load_state = st.info(f"Loading model '{selected_model}'...")
    model_load_state.empty()
    if show_model_info:
        with st.sidebar.container():
            desc = f"""<p style="font-family: 'Tinos', serif;">
                {models[selected_model]["description"]}</p>"""
            st.sidebar.subheader("Model info")
            st.sidebar.markdown(desc, unsafe_allow_html=True)
            st.sidebar.markdown("""---""")
        if selected_model != "Select Model":
            with st.sidebar.container():
                st.sidebar.subheader("Reference")
                desc = f"""<p style="font-family: 'Tinos', serif;">
                    {models[selected_model]["citation"]}</p>"""
                st.sidebar.markdown(desc, unsafe_allow_html=True)
            with st.sidebar.expander("See BibTeX"):
                st.text(models[selected_model]["bibtex"])

            desc = f"""<a style="font-family: 'Tinos', serif;" href="{models[selected_model]["paper"]}">The paper is available here.</a>"""
            st.sidebar.markdown(desc, unsafe_allow_html=True)
            st.sidebar.markdown("""---""")

            with st.sidebar.container():
                st.sidebar.subheader("Code")
                desc = f"""<a style="font-family: 'Tinos', serif;" href="{models[selected_model]["code"]}">Check out Github page</a>"""
                st.sidebar.markdown(desc, unsafe_allow_html=True)
                st.sidebar.markdown("""---""")

        st.sidebar.button("Contact", on_click=show_contacts)
        st.sidebar.button("Other Resources", on_click=others)
        docx_file = st.sidebar.file_uploader("Upload a Text File", type=["txt"])
        thread = Thread(
            target=analyse, args=(docx_file, selected_model, models[selected_model])
        )
        thread = ReportThread.add_report_ctx(thread)
        thread.start()
        thread.join()

        st.sidebar.markdown(f'<form action="{rest_url}"><input style="background-color:#bde9ec" type="submit" value="Go to RestAPI" /></form>', unsafe_allow_html=True)


def analyse(file, selected_model, model):
    if file is not None:
        file_name = file.name.strip(".txt") + "_" + selected_model + "_processed.txt"
        file_write = open(file_name, "w")
        result = ""
        count = 1
        stringio = StringIO(file.getvalue().decode("utf-8"))
        for line in stringio.read().splitlines():
            doc = process_text(selected_model, model, line.strip("\n"))

            if selected_model == "JointModel-DependencyParsing":
                result += "# " + line + "\n" + doc.dep_conll + "\n"
            elif selected_model == "JointModel-MorphemeSegmentation":
                result += "# " + line + "\n" + doc.morph_conll + "\n"
            elif selected_model == "JointModel-MorphemeTagging":
                result += "# " + line + "\n" + doc.morph_tag_conll + "\n"
            elif selected_model == "JointModel-PoSTagging":
                result += "# " + line + "\n" + doc.pos_conll + "\n"
            elif selected_model == "JointModel-All":
                result += "# " + line + "\n" + doc.joint_conll + "\n"
            elif selected_model == "Stemmer":
                result += "# " + line + "\n" + doc.stem_conll + "\n"
            elif selected_model == "NER":
                result += "# " + line + "\n" + doc.ner + "\n"
            elif selected_model == "SemanticParser":
                os.makedirs(file.name.strip(".txt"), exist_ok=True)
                file_write = open(
                    os.path.join(file.name.strip(".txt"), str(count) + ".xml"), "w"
                )
                file_write.write(doc.ucca_conll)
                count += 1
        file_write.write(result)
        # st.sidebar.download_button(label="Analyse", data=result, file_name=file.name.strip(".txt") + "_" + selected_model+ '_processed.txt')


def visualize_parser(
    doc,
    *,
    title: Optional[str] = "Dependency Parse & Part-of-speech tags",
) -> None:
    """Visualizer for dependency parses."""
    if title:
        st.header(title)
    cols = st.columns(4)

    options = {}
    html = displacy.render(doc, options=options, style="dep", manual=True)
    html = html.replace("\n\n", "\n")
    st.write(get_svg(html, "max-width: 100%;"), unsafe_allow_html=True)


def visualize_ner(
    doc,
    *,
    title: Optional[str] = "Named Entities",
    manual: Optional[bool] = True,
) -> None:
    """Visualizer for named entities."""
    if title:
        st.header(title)

    html = displacy.render(
        doc,
        style="ent",
        # options={"ents": label_select, "colors": colors},
        manual=manual,
    )
    style = "<style>mark.entity { display: inline-block }</style>"
    st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)


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
