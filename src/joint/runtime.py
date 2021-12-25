import pickle

from .learner import jPosDepLearner
import pandas as pd


def load_model(model_path, model_opt_path):
    with open(model_opt_path, "rb") as paramsfp:
        words, w2i, c2i, m2i, t2i, morph_dict, pos, rels, stored_opt = pickle.load(
            paramsfp
        )
        stored_opt.external_embedding = None

    print("Loading pre-trained model")
    parser = jPosDepLearner(
        words, pos, rels, w2i, c2i, m2i, t2i, morph_dict, stored_opt
    )
    parser.Load(model_path)
    return parser


def predict_dependecy(model, sentence):
    doc = {"words": [], "arcs": []}
    for entry in model.predict_sentence(sentence):
        if entry.form == "*root*":
            continue
        doc["words"].append({"text": entry.form, "tag": entry.pred_pos})
        if entry.pred_relation == "root":
            continue
        start = entry.id - 1
        end = entry.pred_parent_id - 1
        dir = "left"
        if start > end:
            start, end = end, start
            dir = "right"
        doc["arcs"].append(
            {"start": start, "end": end, "label": entry.pred_relation, "dir": dir}
        )
    return doc


def predict_morphemes(model, sentence):
    doc = {"token": [], "morphemes": []}

    tokens, morhps = model.predict_morphemes(sentence)
    for entry in zip(tokens, morhps):
        if entry[0] == "*root*":
            continue
        doc["token"].append(entry[0])
        doc["morphemes"].append(f"{{{', '.join(entry[1])}}}")
    return doc


def predict_morpheme_tags(model, sentence):
    doc = {"token": [], "morpheme_tags": []}

    tokens, morhps = model.predict_morpheme_tags(sentence)
    for entry in zip(tokens, morhps):
        if entry[0] == "*root*":
            continue
        doc["token"].append(entry[0])
        doc["morpheme_tags"].append(f"{{{', '.join(entry[1][1:-1])}}}")
    return doc
