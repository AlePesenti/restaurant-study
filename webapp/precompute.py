"""Pre-render every sentence the dialog system can say, for every voice condition.

Kokoro needs a few seconds per sentence on a CPU while eSpeak is instant, and that
difference in waiting time would bias the comparison. The dialog system only has a
finite set of responses (templates x restaurants x ontology values), so all of them
are synthesized ahead of time and stored in webapp/audio_cache/. At run time every
condition then answers equally fast.

Run from the repository root (only missing files are generated):
    python -m webapp.precompute            # all voices
    python -m webapp.precompute --list     # only print the sentences
    python -m webapp.precompute --prefetch # only download the models (used by the Dockerfile)
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from ontology import areas, consequents, foods, price_ranges  # noqa: E402
from reasoning import restaurants  # noqa: E402
from responses import contradictions, response_generation, responses  # noqa: E402


GREETING = "Hello, welcome to the Cambridge restaurant recommendation system. How may I help you?"


def all_sentences():
    sentences = {GREETING, *responses.values(), response_generation("askadditionalrequirements")}

    for value in [*areas, *foods, *price_ranges, "dontcare"]:
        sentences.add(response_generation("askconfirmation", suggested_value=value))

    for consequent in consequents:
        sentences.add(response_generation("nomatch2", consequent=consequent))

    for restaurant in restaurants.to_dict("records"):
        sentences.add(response_generation("recommendrestaurant", restaurant=restaurant))

        for consequent in consequents:
            sentences.add(response_generation("recommendrestaurant2", restaurant=restaurant,
                                              consequent=consequent, contradiction=False))
            if consequent in contradictions:
                sentences.add(response_generation("recommendrestaurant2", restaurant=restaurant,
                                                  consequent=consequent, contradiction=True))

        for requested in ["phone", "addr", "postcode", "food", "area", "pricerange"]:
            sentences.add(response_generation("provide", restaurant=restaurant, requested=requested))

        for slot in ["food", "area", "pricerange"]:
            # "Yes, that's correct." or "No, the <slot> is actually <value>."
            sentences.add(response_generation("confirmproperty", restaurant=restaurant,
                                              requested=slot, suggested_value=None))

    sentences.add("Yes, that's correct.")
    return sorted(sentence for sentence in sentences if isinstance(sentence, str) and sentence)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true", help="print the sentences and exit")
    parser.add_argument("--prefetch", action="store_true", help="load every engine and voice once, cache nothing")
    arguments = parser.parse_args()

    sentences = all_sentences()
    if arguments.list:
        print("\n".join(sentences))
        print(f"\n{len(sentences)} sentences")
        return

    import json
    from webapp.tts import SpeechSynthesizer

    conditions = json.loads((ROOT / "webapp" / "study_config.json").read_text(encoding="utf-8"))["conditions"]
    speech = SpeechSynthesizer()

    if arguments.prefetch:
        for voice in conditions.values():
            options = {k: v for k, v in voice.items() if k not in ("engine", "voice", "label")}
            speech._engine(voice["engine"]).synthesize("Hello.", voice["voice"], **options)
        print("models ready")
        return

    for code, voice in conditions.items():
        missing = [text for text in sentences if not speech.on_disk(voice, text)]
        print(f"{code} ({voice['label']}): {len(sentences) - len(missing)} cached, {len(missing)} to render")
        start = time.time()
        for index, text in enumerate(missing, 1):
            speech.audio(voice, text)
            if index % 50 == 0 or index == len(missing):
                elapsed = time.time() - start
                print(f"  {index}/{len(missing)}  {elapsed / index:.2f}s per sentence", flush=True)


if __name__ == "__main__":
    main()
