""" Encoding utterances into frozen DistilBERT sentence embedding
Since the encoder doesn't learn from data we can safely encode the dataset before any spilt (train/data)
We encode solely the unique utterances (4,823 out of 23,998 rows) since it's cheaper and it would give idential vectors for repeated sentences
we run it once to produce embeddings.npz"""

import numpy as np
from sentence_transformers import SentenceTransformer

DATA_FILE = "Data/dailog_acts.dat"
CACHE_FILE = "Data/embeddings.npz"
MODEL_NAME = "distilbert-base-uncased"


def load_and_preprocess_data(file_path):

    dialog_acts = []
    utterances = []

    with open(file_path, encoding="utf-8") as data_raw:
        for line in data_raw:
            line = line.strip().lower()
            dialog_act, utterance = line.split(" ", 1)
            dialog_acts.append(dialog_act)
            utterances.append(utterance)

    return(dialog_acts, utterances)

def encode_unique_utterances(utterances):
    """Encode distinct utterance and return a sentence-vector dictionary"""
    model = SentenceTransformer(MODEL_NAME)
    model.eval() #freeze

    unique_utterances = sorted(set(utterances))
    vectors = model.encode(unique_utterances, batch_size=64, show_progress_bar=True)

    return dict(zip(unique_utterances, vectors))

def save_cache(cache, file_path):
    """store the embeddings as a compressed archive"""
    sentences = list(cache)
    matrix = np.array([cache[sentence] for sentence in sentences], dtype=np.float32)
    np.savez_compressed(file_path, sentences=np.array(sentences), vectors=matrix)

    return matrix.shape


def load_cache(file_path):
    """Load the embedding saved before"""
    archive = np.load(file_path, allow_pickle=False)
    return dict(zip(archive["sentences"], archive["vectors"]))

def main():

    dialog_acts, utterances = load_and_preprocess_data(DATA_FILE)
    print(f"{len(utterances)} rows, {len(set(utterances))} unique utterances")

    cache = encode_unique_utterances(utterances)
    number_of_vectors, dimensions = save_cache(cache, CACHE_FILE)
    
    print(f"saved {number_of_vectors} vectors of {dimensions} dimensions to {CACHE_FILE}")

if __name__ == "__main__":
    main()
