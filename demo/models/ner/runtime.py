from .learner import NERModel
from .config import Config


def load_model(model_path, model_opts_path):
    config = Config(model_opts_path)
    model = NERModel(config)
    model.build()
    model.restore_session(model_path)
    return model


def predict_ner(model, sentence):
    doc = {"text": sentence, "ents": [], "title": None}
    result = ""
    tokens = sentence.split()
    preds = model.predict_ner(tokens)
    start = 0
    for t, p in zip(tokens, preds):
        if p != "O":
            doc["ents"].append({"start": start, "end": start + len(t), "label": p})
            result += str(doc["ents"]) + "\n"
        start += len(t) + 1
    return {"doc": doc, "conllu": result, "sentence": sentence}
