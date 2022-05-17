import pickle

from .learner import jPosDepLearner


def load_model(model_path, model_opt_path):
    with open(model_opt_path, "rb") as paramsfp:
        words, w2i, c2i, m2i, t2i, morph_dict, pos, rels, stored_opt = pickle.load(
            paramsfp
        )
        stored_opt.external_embedding = None

    print("Loading pre-trained parser model")
    parser = jPosDepLearner(
        words, pos, rels, w2i, c2i, m2i, t2i, morph_dict, stored_opt
    )
    parser.Load(model_path)
    return parser


def predict_dependecy_file(model, sentence):
    result = ""
    doc = {"words": [], "arcs": []}
    for entry in model.predict_sentence(sentence):
        if entry.form == "*root*":
            continue
        entry.pred_pos = None
        entry.xpos = None
        entry.pred_tags_tokens = None
        entry.feats = None
        result += str(entry) + "\n"
        
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
        
        
    return {"doc": doc, "conllu": result, "sentence": sentence}

def predict_morphemes_file(model, sentence):
    doc = {}
    doc_1 = {"token": [], "morphemes": []}
    tokens, morhps = model.predict_morphemes(sentence)
    for entry, morph in zip(tokens, morhps):
        if entry == "*root*":
            continue
        doc[entry] = morph
        doc_1["token"].append(entry)
        doc_1["morphemes"].append(f"{{{', '.join(morph)}}}")
    return {"doc": doc_1, "sentence": sentence, "morphemes": doc}

def predict_pos_file(model, sentence):
    result = ""
    doc = {"token": [], "pos": []}
    for entry in model.predict_sentence(sentence):
        if entry.form == "*root*":
            continue
        entry.pred_parent_id = None
        entry.pred_relation = None
        entry.xpos = None
        entry.pred_tags_tokens = None
        entry.feats = None
        result += str(entry) + "\n"
        doc["token"].append(entry.form)
        doc["pos"].append(entry.pred_pos)
    return {"doc": doc, "conllu": result, "sentence": sentence}


def predict_morpheme_tags_file(model, sentence):
    result = ""
    doc = {"token": [], "morpheme_tags": []}
    for entry in model.predict_sentence(sentence):
        if entry.form == "*root*":
            continue
        entry.pred_parent_id = None
        entry.pred_relation = None
        entry.pred_pos = None
        entry.xpos = None
        result += str(entry) + "\n"
        doc["token"].append(entry.form)
        doc["morpheme_tags"].append(f"{{{', '.join(entry.pred_tags_tokens)}}}")
    return {"doc" : doc, "conllu": result, "sentence": sentence}