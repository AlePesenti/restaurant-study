from sklearn.model_selection import train_test_split

from preprocess import load_and_preprocess_data
from evaluation import evaluate
from classifiers import rule_based_classifier


RANDOM_SEED = 42
DATA_PATH = "Data/dailog_acts.dat"

dialog_acts, utterances = load_and_preprocess_data(DATA_PATH)

# 1. Original split
X_train, X_test, y_train, y_test = train_test_split(utterances, dialog_acts, test_size=0.15, stratify=dialog_acts, random_state=RANDOM_SEED)

# 2. Grouped split
from preprocess import grouped_split

X_train_grouped, X_test_grouped, y_train_grouped, y_test_grouped = grouped_split(utterances, dialog_acts)


##### CLASSIFIERS
### 1. Rule-based baseline
rule_based_predictions = rule_based_classifier(X_test)


### 2. Train two ML classifiers on a bag-of-words representation of the utterances.

# Create bag-of-words representation
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()

X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow  = vectorizer.transform(X_test)


vectorizer_grouped = CountVectorizer()

X_train_grouped_bow = vectorizer_grouped.fit_transform(X_train_grouped)
X_test_grouped_bow  = vectorizer_grouped.transform(X_test_grouped)


# 1. Logistic regression
from classifiers import log_reg

log_reg_predictions = log_reg(X_train_bow, y_train, X_test_bow)

log_reg_grouped_predictions = log_reg(X_train_grouped_bow, y_train_grouped, X_test_grouped_bow)



# 2. SVM
from classifiers import svm

svm_predictions = svm(X_train_bow, y_train, X_test_bow)

svm_grouped_predictions = svm(X_train_grouped_bow, y_train_grouped, X_test_grouped_bow)


### 3. Train the same two ML classifiers on top of frozen pretrained embeddings.

from embeddings import load_cache
import numpy as np

cache = load_cache("Data/embeddings.npz")
X_train_emb = np.array([cache[u] for u in X_train])
X_test_emb = np.array([cache[u] for u in X_test])

X_train_grouped_emb = np.array([cache[u] for u in X_train_grouped]) 
X_test_grouped_emb = np.array([cache[u] for u in X_test_grouped])

# 1. Logistic regression

log_reg_predictions_emb = log_reg(X_train_emb, y_train, X_test_emb)

log_reg_grouped_predictions_emb = log_reg(X_train_grouped_emb, y_train_grouped, X_test_grouped_emb)


# 2. SVM

svm_predictions_emb = svm(X_train_emb, y_train, X_test_emb)

svm_grouped_predictions_emb = svm(X_train_grouped_emb, y_train_grouped, X_test_grouped_emb)


##### EVALUATION

print("Rule-based baseline")
evaluate(y_test, rule_based_predictions)


print("BoW + logistic regression")
evaluate(y_test, log_reg_predictions)

print("BoW + logistic regression - Grouped")
evaluate(y_test_grouped, log_reg_grouped_predictions)


print("BoW + SVM")
evaluate(y_test, svm_predictions)

print("BoW + SVM - Grouped")
evaluate(y_test_grouped, svm_grouped_predictions)


print("Embeddings + logistic regression")
evaluate(y_test, log_reg_predictions_emb)

print("Embeddings + logistic regression - Grouped")
evaluate(y_test_grouped, log_reg_grouped_predictions_emb)


print("Embeddings + SVM")
evaluate(y_test, svm_predictions_emb)

print("Embeddings + SVM - Grouped")
evaluate(y_test_grouped, svm_grouped_predictions_emb)