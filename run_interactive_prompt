
"""Interactive Prompt Script"""

import os

import joblib

from classifiers import rule_based_baseline

MODEL_FILE = "models/bow_models.joblib"
CLASSIFIERS = ("baseline", "log_reg", "svm")


def predict(utterance, classifier_name, models):
    #Return the predicted act for single utterance
    if classifier_name == "baseline":
        return rule_based_baseline(utterance)

    features = models["vectorizer"].transform([utterance])

    return models[classifier_name].predict(features)[0]


def main():
    if not os.path.exists(MODEL_FILE):
        print("Model file not found:", MODEL_FILE)
        print("Run main.py to train classifieras")
        return

    models = joblib.load(MODEL_FILE)
    classifier_name = "svm"

    print("Dialog act classifier, the current classifier is :", classifier_name)
    print("Type an utterance, or", " / ".join(CLASSIFIERS), "to switch classifier, or exit to quit.")

    while True:
        # handled errors on edge cases( empty input, punctuation, and unusual characters)
        try:
            #input is converted to lower case as asked in the assignement
            utterance = input("\n> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break

        if utterance == "exit":
            break

        if not utterance:
            continue

        if utterance in CLASSIFIERS:
            classifier_name = utterance
            print("new classifier :", classifier_name)
            continue

        print(classifier_name, ":  ", predict(utterance, classifier_name, models))

if __name__ == "__main__":
    main()