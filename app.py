from pathlib import Path

import srsly

from visualizer import visualize

MODELS = srsly.read_json(Path(__file__).parent / "models.json")
DEFAULT_MODEL = "JointModel-All"
DEFAULT_TEXT = "Ankara'nın Çankaya ilçesinde kontrolden çıkan bir otomobil yol kenarındaki kayalıklara takılarak askıda kaldı"
DESCRIPTION = """**Explore trained [spaCy v3.0](https://nightly.spacy.io) pipelines**"""


visualize(
    MODELS,
    default_model=DEFAULT_MODEL,
    default_text=DEFAULT_TEXT,
    sidebar_title="Turkish NLP Toolkit",
)
