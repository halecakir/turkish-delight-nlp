import os

import srsly


DEFAULT_MODEL = "JointModel-All"
DESCRIPTION = """**Explore trained [spaCy v3.0](https://nightly.spacy.io) pipelines**"""

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
PAPERS_MD = os.environ["PAPERS_MD"]
CONTACT_MD = os.environ["CONTACT_MD"]
MODELS = srsly.read_json(os.environ["DEFAULT_MODELS"])
DATA_ROOT = os.environ["DATA_ROOT"]
REST_URL=os.environ["REST_URL"]