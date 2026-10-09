"""Evaluate classifiers on a held-out file
    Result are saved to results/heldout.txt"""
import os
import sys

import joblib

from preprocess import load_and_preprocess_data
from evaluation import evaluate

BOW_MODELS = "models/bow_models.joblib"
EMBEDDING_MODELS = "models/embedding_models.joblib"
DEFAULT_TEST_FILE = "Data/dialog_acts_test.dat"
RESULTS_FILE = "results/heldout.txt"

def build_features(utterances, use_embeddings):
    #convert the utterances in the matrix for the model"
    if use_embeddings:
        from embeddings import encode_unique_utterances
        cache = encode_unique_utterances(utterances)
        models = joblib.load(EMBEDDING_MODELS)

        return [cache[utterance] for utterance in utterances], models

    models = joblib.load(BOW_MODELS)
    return models["vectorizer"].transform(utterances), models

def main():
    arguments = sys.argv[1:]
    use_embeddings= "--emnbb" in arguments
    paths = [argument for argument in arguments if not argument.startswith("--")]
    test_file= paths[0] if paths else DEFAULT_TEST_FILE

    if not os.path.exists(test_file):
        print("Test file not found:", test_file)
        print("Pass the path as an argument, in the right location")
        return

    model_file = EMBEDDING_MODELS if use_embeddings else BOW_MODELS

    if not os.path.exists(model_file):
        print("Model file not found:", model_file)
        print("Run main.py to train classifieras")
        return

    dialog_acts, utterances = load_and_preprocess_data(test_file)
    features, models = build_features(utterances, use_embeddings)

    print(len(utterances), "utterances loaded from", test_file)

    os.makedirs("results", exist_ok=True)

    with open(RESULTS_FILE, "w", encoding="utf-8") as results:
        results.write("Held-out evaluation of " + test_file + "\n")
        results.write(("embedding" if use_embeddings else "bag-of-words") + " models\n\n")

        for name in ("log_reg", "svm"):
            predictions = models[name].predict(features)

            print("\n" + name)
            scores = evaluate(dialog_acts, predictions)

            results.write(name + "\n")
            for metric, value in scores.items():
                results.write("  " + metric + ": " + str(value) + "\n")
            results.write("\n")

    print("\nResults written to", RESULTS_FILE)


if __name__ == "__main__":
    main()