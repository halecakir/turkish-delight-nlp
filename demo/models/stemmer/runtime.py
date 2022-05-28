from .learner import Stemmer


def load_model(model_path):
    stemmer = Stemmer()
    print("Loading pre-trained stemmer model")
    stemmer.load(model_path)
    return stemmer


def predict_stems(model, document):
    doc = {"words": [], "stems": []}
    result = ""
    for word in document.split():
        word = word.lower()
        doc["words"].append(word)
        stemmed_word = model.predict_stem(word)
        doc["stems"].append(stemmed_word)
        result += word + "\t" + stemmed_word + "\n"
    return {"doc": doc, "conllu": result}
