"""Fixes to the Part 1b dialog system, applied only in the web app.

The submitted system (state_machine.py, slot_extraction.py) is left untouched; this
module wraps it. Problems found in the pilot conversations:

1. Keywords were matched as substrings, so "inexpensive" was read as "expensive"
   (and e.g. "at least" as "east"). Here they are matched as whole words, and the typo
   fallback no longer "corrects" ordinary words such as "eat" (east) or "that" (thai).
2. Common synonyms were not understood ("central", "northern", "affordable", ...).
   They are rewritten to the ontology values before slot extraction.
3. Preferences were only extracted when the classifier said "inform". When the system
   has just asked for food, area or price, an answer classified as "request" or "null"
   ("close to the central area", "nothern") is now still searched for preferences.
4. After a recommendation, "no" or an unknown request restarted the lookup and looped
   back to "Do you have any additional requirements?". Now "no" gives an alternative,
   "what is its name" repeats the recommendation, and an unclear turn asks to rephrase
   instead of losing the recommendation.

Every answer is still one of the system's own sentences, so all audio stays pre-rendered.
"""

import copy
import re

import state_machine
import slot_extraction
from ontology import areas, consequents, foods, price_ranges
from responses import response_generation

original_keyword_slot_extractor = slot_extraction.keyword_slot_extractor
original_levenshtein_slot_extractor = slot_extraction.levenshtein_slot_extractor

# Longer phrases first, so "not expensive" is rewritten before "expensive" is seen.
SYNONYMS = [
    (r"not (?:too |very |that |so )?expensive|inexpensive|affordable|budget|cheaply|cheaper|"
     r"low[- ]cost|low price[ds]?|low[- ]priced|reasonably cheap", "cheap"),
    (r"moderately priced|moderately|mid[- ]range|mid[- ]priced|medium[- ]priced|medium price|"
     r"average price[ds]?|average[- ]priced|reasonably priced|reasonable price[ds]?|reasonable", "moderate"),
    (r"pricey|pricy|upscale|high[- ]end|luxury|luxurious|costly|fancy", "expensive"),
    (r"city cent(?:re|er)|town cent(?:re|er)|central|center|downtown|middle of (?:the )?town", "centre"),
    (r"northern|nothern|norht|nort", "north"),
    (r"southern|souther|sotuh", "south"),
    (r"eastern|esat", "east"),
    (r"western|wset", "west"),
]

# Only meaningful as an answer to "What is your preferred price range?"
PRICE_ANSWERS = [
    (r"not (?:too )?much|not a lot|as little as possible|low", "cheap"),
    (r"a lot|whatever it costs|high", "expensive"),
]

SLOT_STATES = {3: "food", 4: "area", 5: "pricerange"}
NEVER_OVERRIDE = {"bye", "thankyou", "restart", "repeat"}


def normalize(utterance, current_state=None):
    utterance = utterance.lower()
    rules = SYNONYMS + (PRICE_ANSWERS if current_state == 5 else [])
    for pattern, value in rules:
        utterance = re.sub(rf"\b(?:{pattern})\b", value, utterance)
    return utterance


def whole_word(value, utterance):
    return re.search(rf"\b{re.escape(value)}\b", utterance) is not None


def keyword_slot_extractor(utterance, current_state):
    utterance = normalize(utterance, current_state)
    # The original handles the "dontcare" phrases; its value matches are redone below.
    slots = original_keyword_slot_extractor(utterance, current_state)

    for slot, values in [("area", areas), ("food", foods), ("pricerange", price_ranges),
                         ("consequent", consequents)]:
        if slots[slot] == "dontcare":
            continue
        matches = [value for value in values if whole_word(value, utterance)]
        slots[slot] = max(matches, key=len) if matches else None

    return slots


# Ordinary words one edit away from an ontology value ("eat" -> east, "that" -> thai),
# which the typo fallback would otherwise "correct" and ask to confirm.
NOT_TYPOS = {"eat", "best", "rest", "test", "fast", "last", "least", "feast", "beast", "vest",
             "nest", "that", "than", "tai"}


def levenshtein_slot_extractor(utterance, threshold=1):
    words = [word for word in normalize(utterance).split() if word not in NOT_TYPOS]
    return original_levenshtein_slot_extractor(" ".join(words), threshold)


# The state machine imported these names directly, so they are replaced there
# (only inside the web app process).
state_machine.keyword_slot_extractor = keyword_slot_extractor
state_machine.levenshtein_slot_extractor = levenshtein_slot_extractor


def mentions_preference(utterance, current_state):
    slots = keyword_slot_extractor(utterance, current_state)
    if any(slots[slot] for slot in ("area", "food", "pricerange")):
        return True
    fallback = levenshtein_slot_extractor(utterance)
    return any(fallback.values())


def transition(dialog_state, utterance, dialog_act):
    """Same interface as state_machine.state_transition; returns (state, response, act used)."""
    current_state = dialog_state["state"]
    act = dialog_act

    # 3. Answers to a slot question are searched for preferences whatever the dialog act.
    if (current_state in (1, 2, 3, 4, 5, 7) and act not in ("inform", "reqalts")
            and act not in NEVER_OVERRIDE and mentions_preference(utterance, current_state)):
        act = "inform"

    restaurant = dialog_state["restaurant"]

    # 4. After a recommendation (or while asking for additional requirements).
    if current_state in (8, 9, 10) and restaurant is not None:
        if act == "request" and re.search(r"\bname\b|\bcalled\b|\bwhich (?:one|restaurant)\b", utterance):
            dialog_state["state"] = 8 if current_state == 10 else current_state
            response = response_generation("recommendrestaurant", restaurant=restaurant)
            dialog_state["last_response"] = response
            return dialog_state, response, act

        if current_state == 10 and act == "request":
            # The user asks for details before a restaurant was named: name it first.
            dialog_state["state"] = 8
            response = response_generation("recommendrestaurant", restaurant=restaurant)
            dialog_state["last_response"] = response
            return dialog_state, response, act

        if current_state in (8, 9) and act in ("negate", "deny"):
            # "no thanks, that's all" ends the conversation; a plain "no" asks for another one.
            closing = r"\bthat'?s all\b|\bthat is all\b|\bthanks?\b|\bthank you\b|\bbye\b|\bgoodbye\b"
            act = "bye" if re.search(closing, utterance) else "reqalts"

    before = copy.deepcopy(dialog_state)
    dialog_state, response = state_machine.state_transition(dialog_state, utterance, act)

    # 4. Nothing new was understood after a recommendation: the original would redo the
    # lookup and ask for additional requirements again. Keep the recommendation instead.
    slots_unchanged = all(before[slot] == dialog_state[slot] for slot in ("food", "area", "pricerange"))
    if (current_state in (8, 9) and dialog_state["state"] == 10 and slots_unchanged
            and response == response_generation("askadditionalrequirements")):
        dialog_state = before
        response = response_generation("null")
        dialog_state["last_response"] = response

    return dialog_state, response, act
