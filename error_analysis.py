from collections import Counter

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


def get_classification_report(y_test, y_pred):

    return classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )


def get_confusion_matrix(y_test, y_pred, labels=None):

    return confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )


def find_hard_examples(utterances, y_test, predictions, model_names):

    hard_examples = []

    for i, utterance in enumerate(utterances):

        actual = y_test[i]

        model_predictions = [
            predictions[j][i]
            for j in range(len(predictions))
        ]

        if all(pred != actual for pred in model_predictions):

            hard_examples.append({
                "utterance": utterance,
                "actual": actual,
                "predictions": model_predictions,
                "model_names": model_names
            })

    return hard_examples


def get_common_errors(y_test, y_pred, top_k=15):

    errors = []

    for actual, predicted in zip(y_test, y_pred):

        if actual != predicted:
            errors.append((actual, predicted))

    return Counter(errors).most_common(top_k)


def print_hard_examples(hard_examples):

    for i, example in enumerate(hard_examples):

        print("Utterance:", example["utterance"])
        print("Actual:", example["actual"])

        for model, prediction in zip(
            example["model_names"],
            example["predictions"]
        ):
            print(model + ":", prediction)

        print("----------------------------------------")