from ontology import areas, foods, price_ranges, consequents


### 1. Keyword matching

def keyword_slot_extractor(utterance, current_state):
    utterance = utterance.lower()
    
    # Reference dialog 12
    utterance = utterance.replace("center", "centre")

    slots = {
        "area": None,
        "food": None,
        "pricerange": None,
        "consequent": None
    }
    
    for area in areas:
        if area in utterance:
            slots["area"] = area

    for food in foods:
        if food in utterance:
            slots["food"] = food

    for price in price_ranges:
        if price in utterance:
            slots["pricerange"] = price
            
    for consequent in consequents:
        if consequent in utterance:
            slots["consequent"] = consequent
            
    ### Dontcare part (If the user doesnt have a preference), this maybe has to be made more clean/efficient or different
    ### I do not think it is supposed to be this hard-coded?
    
    if "any area" in utterance or "any part of town" in utterance or "any location" in utterance:
        slots["area"] = "dontcare"

    if "any price" in utterance:
        slots["pricerange"] = "dontcare"

    if "any food" in utterance:
        slots["food"] = "dontcare"
        

    if (utterance.strip() == "any" 
        or "i don't care" in utterance
        or "i dont care" in utterance 
        or "dont care" in utterance
        or "it doesn't matter" in utterance
        or "it doesnt matter" in utterance):

        if current_state == 3:
            slots["food"] = "dontcare"

        elif current_state == 4:
            slots["area"] = "dontcare"

        elif current_state == 5:
            slots["pricerange"] = "dontcare"
            
            
            
    
    return slots

### 2. Levenshtein edit distance

#%pip install levenshtein
from Levenshtein import distance

def levenshtein_slot_extractor(utterance, threshold=1):  # Maybe the threshold is better at 2
    utterance = utterance.lower()
    words = utterance.split()

    slots = {
        "area": None,
        "food": None,
        "pricerange": None
    }
    
    # Area
    best_match = None
    best_distance = float("inf")

    for area in areas:
        for word in words:
            current_distance = distance(word, area)

            if current_distance < best_distance:
                best_distance = current_distance
                best_match = area

    if best_distance <= threshold:
        slots["area"] = best_match        
    
    
    # Food
    best_match = None
    best_distance = float("inf")

    for food in foods:
        for word in words:
            current_distance = distance(word, food)

            if current_distance < best_distance:
                best_distance = current_distance
                best_match = food

    if best_distance <= threshold:
        slots["food"] = best_match
    

    # Pricerange
    best_match = None
    best_distance = float("inf")

    for price in price_ranges:
        for word in words:
            current_distance = distance(word, price)

            if current_distance < best_distance:
                best_distance = current_distance
                best_match = price

    if best_distance <= threshold:
        slots["pricerange"] = best_match

    return slots

### 3. Semantic similarity via embeddings

import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings import load_cache

SEMANTIC_MODEL = "distilbert-base-uncased"
SIMILARITY_THRESHOLD = 0.15

#Words that doesn't describe a preference need to be removed. The word domanin words outscore the real cue.
GENERIC_WORDS = {"restaurant", "restaurants", "place", "food", "town", "area",
                 "part", "price", "range", "looking", "want", "like", "would",
                 "something", "please", "serving", "serves", "eat", "the",
                 "for", "any", "and"}

MINIMUM_WORD_LENGTH = 3

semantic_model = SentenceTransformer(SEMANTIC_MODEL)
semantic_model.eval()   # frozen:used only as a fixed feature extractor

#DistilBERT similarities are nearly uniform. Whitening fixes this issue (We need to adress it in the report). 
cached_vectors = np.array(list(load_cache("Data/embeddings.npz").values()),
                          dtype=np.float64)
embedding_mean = cached_vectors.mean(axis=0, keepdims=True)
covariance = np.cov(cached_vectors.T)
components, variances, _ = np.linalg.svd(covariance)
whitening_matrix = components @ np.diag(1.0 / np.sqrt(variances + 1e-8))


def prepare(vectors):
    #Whitening and dividing each vector by its own lenght. We then use dot product instead of cosine similarity
    whitened = (np.atleast_2d(vectors).astype(np.float64) - embedding_mean) @ whitening_matrix
    norms = np.linalg.norm(whitened, axis=1, keepdims=True)
    return whitened / np.clip(norms, 1e-8, np.inf)


ontology = {"area": areas, "food": foods, "pricerange": price_ranges}

#encoded once
ontology_vectors = {slot: prepare(semantic_model.encode(terms))
                    for slot, terms in ontology.items()}


def semantic_slot_extractor(utterance, threshold=SIMILARITY_THRESHOLD, debug=False):
    #Map the words of utterances to ontology by cosine similarity
    #Matches below the threshold are rejecte
    words = [word for word in utterance.lower().split()
             if len(word) >= MINIMUM_WORD_LENGTH and word not in GENERIC_WORDS]

    slots = {"area": None, "food": None, "pricerange": None}

    if not words:
        return slots

    best_similarity = {slot: 0.0 for slot in ontology}

    for word, word_vector in zip(words, prepare(semantic_model.encode(words))):
        candidates = []

        for slot, terms in ontology.items():
            similarities = word_vector @ ontology_vectors[slot].T
            best = int(similarities.argmax())
            candidates.append((float(similarities[best]), slot, terms[best]))

        similarity, slot, term = max(candidates)

        if debug:
            verdict = "accepted" if similarity >= threshold else "rejected"
            print(f" {word} -> {slot} {term} {similarity}  {verdict}")

        if similarity >= threshold and similarity > best_similarity[slot]:
            best_similarity[slot] = similarity
            slots[slot] = term

    return slots

def request_extractor(utterance):
    utterance = utterance.lower()

    if "phone" in utterance:
        return "phone"

    elif "address" in utterance:
        return "addr"

    elif "postcode" in utterance or "post code" in utterance or "postal code" in utterance:
        return "postcode"

    elif "food" in utterance:
        return "food"

    elif "area" in utterance:
        return "area"

    elif "price" in utterance:
        return "pricerange"
