import argparse

from state_machine import create_dialog_state, state_transition


### Dialog act classifier, loaded from Part 1a ###
# Trained and saved by main.py. Loading instead of retraining 
import joblib

saved_models = joblib.load("models/bow_models.joblib")
dialog_vectorizer = saved_models["vectorizer"]
dialog_classifier = saved_models["svm"]


def classify_utterance(utterance):

    utterance = utterance.strip().lower()

    if utterance == "":
        return "null"

    vector = dialog_vectorizer.transform([utterance])

    return dialog_classifier.predict(vector)[0]

################################################################################################################################

####### Text to speech #####
import pyttsx3
#pyttsx3.speak("I will speak this text")

#!pip install pyttsx3

speech = pyttsx3.init() 

def speak(text):
    """
    pyttsx3.init() reuses the engine it created before, and the Windows SAPI5
    driver stops producing sound after the first runAndWait(). Clearing the
    engine cache forces a genuinely new engine for every utterance.
    """
    pyttsx3._activeEngines.clear()

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    voices = engine.getProperty("voices")
    engine.setProperty("voice", voices[1].id)
    engine.say(text)
    engine.runAndWait()
######################################


### Interaction logging for the Part 2 analysis
# One JSON object per turn, appended to logs/dialogs.jsonl.
import json
import os
import uuid
from datetime import datetime

LOG_FILE = "logs/dialogs.jsonl"


def log_turn(record):
    """Append one turn to the log file, creating the folder on first use."""
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    with open(LOG_FILE, "a", encoding="utf-8") as log:
        log.write(json.dumps(record) + "\n")


def run_dialog(debug=False, TTS=False, fallback="levenshtein"):
    if fallback not in ("levenshtein", "semantic"):
        raise ValueError('fallback must be "levenshtein" or "semantic", got: ' + repr(fallback))
    dialog_state = create_dialog_state()

    session_id = str(uuid.uuid4())[:8]
    turn = 0

    print("System: Hello, welcome to the Cambridge restaurant recommendation system. How may I help you?")    
    
    ### Text to speech
    if TTS:
        speak("Hello, welcome to the Cambridge restaurant recommendation system. How may I help you?")
    
    while dialog_state["state"] != 11:

        utterance = input("User: ").strip().lower()

        dialog_act = classify_utterance(utterance)

        if debug:
            print(f"[Predicted dialog act: {dialog_act}]")
        
        state_before = dialog_state["state"]
        dialog_state, response = state_transition(dialog_state, utterance, dialog_act, fallback=fallback)

        print("System:", response)

        turn += 1
        log_turn({
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "tts": TTS,
            "fallback": fallback,
            "turn": turn,
            "state_before": state_before,
            "user_utterance": utterance,
            "dialog_act": dialog_act,
            "state_after": dialog_state["state"],
            "system_response": response,
            "area": dialog_state["area"],
            "food": dialog_state["food"],
            "pricerange": dialog_state["pricerange"],
            "recommendation": dialog_state["restaurant"],
        })
        
        ### Text to speech
        if TTS:
            speak(response)

    return dialog_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tts", action="store_true",
                        help="speak the system utterances out loud")
    parser.add_argument("--fallback", default="levenshtein",
                        choices=["levenshtein", "semantic"],
                        help="which slot extraction fallback to use")
    parser.add_argument("--debug", action="store_true",
                        help="print the predicted dialog act for each turn")
    arguments = parser.parse_args()

    run_dialog(debug=arguments.debug, TTS=arguments.tts, fallback=arguments.fallback)


if __name__ == "__main__":
    main()
