from visualizer import visualize
from pathlib import Path
import srsly
import importlib
import random

MODELS = srsly.read_json(Path(__file__).parent / "models.json")
DEFAULT_MODEL = "JointModel"
DEFAULT_TEXT = "Ankara'nın Çankaya ilçesinde kontrolden çıkan bir otomobil yol kenarındaki kayalıklara takılarak askıda kaldı"
#DEFAULT_TEXT= "O eve geldi"
DESCRIPTION = """**Explore trained [spaCy v3.0](https://nightly.spacy.io) pipelines**"""

# def get_default_text(nlp):
#     # Check if spaCy has built-in example texts for the language
#     try:
#         examples = importlib.import_module(f".lang.{nlp.lang}.examples", "spacy")
#         return examples.sentences[0]
#     except (ModuleNotFoundError, ImportError):
#         return ""

visualize(
    MODELS,
    default_model=DEFAULT_MODEL,
    default_text=DEFAULT_TEXT,
    show_visualizer_select=True,
    #sidebar_description="This is the place for desc"
)
