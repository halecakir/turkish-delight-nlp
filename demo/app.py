from conf import CONTACT_MD, DEFAULT_MODEL, MODELS, PAPERS_MD, REST_URL
from visualizer import visualize

visualize(
    MODELS,
    default_model=DEFAULT_MODEL,
    sidebar_title="Turkish Delight NLP",
    papers_md=PAPERS_MD,
    contact_md=CONTACT_MD,
    rest_url=REST_URL
)
