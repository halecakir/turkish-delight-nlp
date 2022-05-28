

from os import environ

import srsly

APP_VERSION = "0.1"
APP_NAME = "Turkish Delight NLP API"

API_PREFIX = environ['API_PREFIX']
API_KEY = environ['API_KEY']
IS_DEBUG =  bool(environ['IS_DEBUG'])
DEFAULT_MODELS =  srsly.read_json(environ['DEFAULT_MODELS'])
DATA_PATH = environ['DATA_ROOT']

print(API_KEY, IS_DEBUG, DEFAULT_MODELS, DATA_PATH)
