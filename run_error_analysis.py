from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

from preprocess import load_and_preprocess_data
from classifiers import svm
from error_analysis import (
    get_classification_report,
    get_confusion_matrix,
    get_common_errors
)


RANDOM_SEED = 42
DATA_PATH = "Data/dailog_acts.dat"


dialog_acts, utterances = load_and_preprocess_data(DATA_PATH)


# Split data

X_train, X_test, y_train, y_test = train_test_split(
    utterances,
    dialog_acts,
    test_size=0.15,
    stratify=dialog_acts,
    random_state=RANDOM_SEED
)


# Bag-of-words

vectorizer = CountVectorizer()

X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)


# Train SVM

svm_predictions = svm(
    X_train_bow,
    y_train,
    X_test_bow
)


##### CLASSIFICATION REPORT

print("Classification report - BoW + SVM")

svm_report = get_classification_report(
    y_test,
    svm_predictions
)

for dialog_act in sorted(set(y_test)):

    print(
        dialog_act,
        "F1:",
        svm_report[dialog_act]["f1-score"],
        "support:",
        svm_report[dialog_act]["support"]
    )


##### CONFUSION MATRIX

labels = sorted(set(y_test))

cm = get_confusion_matrix(
    y_test,
    svm_predictions,
    labels
)

print("\nLabels:")
print(labels)

print("\nConfusion matrix:")
print(cm)


##### MOST COMMON ERRORS

common_svm_errors = get_common_errors(
    y_test,
    svm_predictions,
    top_k=15
)

print("\nMost common SVM errors")

for (actual, predicted), count in common_svm_errors:

    print(
        actual,
        "->",
        predicted,
        ":",
        count
    )


##### SPECIFIC EXAMPLE

target_utterance = "i dont care about the price range what is the address"

print("\nSpecific example")

for i, utterance in enumerate(X_test):

    if utterance.lower() == target_utterance.lower():

        print("Utterance:", utterance)
        print("Actual:", y_test[i])
        print("Prediction:", svm_predictions[i])