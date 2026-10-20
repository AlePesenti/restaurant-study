"""Web app for the TTS perception study.

Each participant talks to the restaurant dialog system twice, once with a robotic and
once with a natural voice of the same gender. The researcher picks the pair and its order
by sending one of the group links (?g=CODE, see "groups" in study_config.json). After
each conversation the participant is sent to the Google Form with their participant ID
and an opaque condition code prefilled.

Run locally from the repository root:
    python -m uvicorn webapp.server:app --port 7860
"""

import base64
import hashlib
import hmac
import json
import os
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from state_machine import create_dialog_state
from webapp import dialog as web_dialog
from webapp.tts import SpeechSynthesizer

ROOT = Path(__file__).resolve().parent.parent
WEBAPP = Path(__file__).resolve().parent

# The dialog system modules open "Data/..." with relative paths.
os.chdir(ROOT)

CONFIG = json.loads((WEBAPP / "study_config.json").read_text(encoding="utf-8"))
CONDITIONS = CONFIG["conditions"]
GROUPS = CONFIG["groups"]
FORM_URL = os.environ.get("FORM_URL", CONFIG["form_url"])
FINAL_FORM_URL = os.environ.get("FINAL_FORM_URL", CONFIG["final_form_url"])

# Signs the session token so that a server restart does not lose a participant's order.
SECRET_KEY = os.environ.get("SECRET_KEY", "change-me-in-the-space-settings").encode("utf-8")

GREETING = "Hello, welcome to the Cambridge restaurant recommendation system. How may I help you?"


### Dialog act classifier (same model as run_dialog_system.py) ###

saved_models = joblib.load(ROOT / "models" / "bow_models.joblib")
dialog_vectorizer = saved_models["vectorizer"]
dialog_classifier = saved_models["svm"]


def classify_utterance(utterance):
    utterance = utterance.strip().lower()

    if utterance == "":
        return "null"

    return dialog_classifier.predict(dialog_vectorizer.transform([utterance]))[0]


### Session tokens ###

def make_token(participant_id, order):
    payload = base64.urlsafe_b64encode(json.dumps([participant_id, order]).encode()).decode()
    signature = hmac.new(SECRET_KEY, payload.encode(), hashlib.sha256).hexdigest()[:24]
    return f"{payload}.{signature}"


def read_token(token):
    try:
        payload, signature = token.split(".")
        expected = hmac.new(SECRET_KEY, payload.encode(), hashlib.sha256).hexdigest()[:24]
        if not hmac.compare_digest(signature, expected):
            raise ValueError
        participant_id, order = json.loads(base64.urlsafe_b64decode(payload))
        return participant_id, order
    except Exception:
        raise HTTPException(status_code=400, detail="invalid session")


### Logging ###
# One JSON object per event. If LOG_DATASET (e.g. "user/tts-study-logs") and HF_TOKEN
# are set, the folder is committed to that private Hugging Face dataset every few
# minutes, because the Space's own disk is wiped on every restart.

LOG_DIR = ROOT / "logs" / "webapp"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"events-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}.jsonl"

log_lock = threading.Lock()
scheduler = None

if os.environ.get("LOG_DATASET") and os.environ.get("HF_TOKEN"):
    from huggingface_hub import CommitScheduler

    scheduler = CommitScheduler(repo_id=os.environ["LOG_DATASET"], repo_type="dataset",
                                folder_path=LOG_DIR, path_in_repo="logs", every=2,
                                private=True, token=os.environ["HF_TOKEN"])
    log_lock = scheduler.lock


def log_event(event, **fields):
    record = {"event": event, "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), **fields}
    with log_lock:
        with open(LOG_FILE, "a", encoding="utf-8") as log:
            log.write(json.dumps(record) + "\n")


### Speech ###

speech = SpeechSynthesizer()


def speak(code, text):
    """URL of the audio for this sentence, or None if it cannot be produced.

    Every sentence is pre-rendered by webapp/precompute.py. If one is missing, the server
    tries to render it, which only works where the TTS engines are installed (not on the
    lightweight hosted server); the participant then sees the text without audio.
    """
    try:
        key, _ = speech.audio(CONDITIONS[code], text)
    except Exception as error:
        print(f"no audio for {code} {text!r}: {error}")
        log_event("missing_audio", code=code, text=text)
        return None
    return f"api/audio/{key}.mp3"


def check_cache():
    from webapp.precompute import all_sentences
    sentences = all_sentences()
    for code, voice in CONDITIONS.items():
        missing = sum(not speech.on_disk(voice, text) for text in sentences)
        print(f"audio cache {code} ({voice['label']}): {len(sentences) - missing}/{len(sentences)} sentences")


### Dialogs in progress, keyed by (participant_id, position) ###

dialogs = {}
dialogs_lock = threading.Lock()


def form_link(template, participant_id, code="", position=""):
    if not template:
        return ""
    group = participant_id.split("-")[0]
    return template.format(participant_id=quote(participant_id), code=quote(code),
                           position=position, group=quote(group))


### API ###

@asynccontextmanager
async def lifespan(app):
    check_cache()
    yield


app = FastAPI(lifespan=lifespan)


class SessionRequest(BaseModel):
    token: str


class NewSessionRequest(BaseModel):
    group: str


class StartRequest(BaseModel):
    token: str
    position: int


class MessageRequest(BaseModel):
    token: str
    position: int
    text: str


@app.get("/")
def index():
    return FileResponse(WEBAPP / "static" / "index.html")


@app.get("/api/config")
def config(g: str = ""):
    group = GROUPS.get(g.strip().upper())
    return {"valid_group": group is not None, "conditions": len(group["order"]) if group else 0,
            "show_text": CONFIG["show_text"], "min_turns_before_done": CONFIG["min_turns_before_done"]}


@app.post("/api/session")
def new_session(request: NewSessionRequest):
    group = request.group.strip().upper()
    if group not in GROUPS:
        raise HTTPException(status_code=400, detail="invalid group")

    # The group code starts the participant ID, so every form response carries it.
    participant_id = f"{group}-{uuid.uuid4().hex[:6].upper()}"
    order = GROUPS[group]["order"]
    log_event("session_start", participant_id=participant_id, group=group,
              group_label=GROUPS[group]["label"], order=order,
              order_labels=[CONDITIONS[code]["label"] for code in order])
    return {"token": make_token(participant_id, order), "participant_id": participant_id}


def condition_for(token, position):
    participant_id, order = read_token(token)
    if not 1 <= position <= len(order):
        raise HTTPException(status_code=400, detail="invalid position")
    return participant_id, order[position - 1]


@app.post("/api/start")
def start(request: StartRequest):
    participant_id, code = condition_for(request.token, request.position)

    with dialogs_lock:
        dialogs[(participant_id, request.position)] = {"state": create_dialog_state(), "turns": 0}

    tasks = CONFIG["tasks"]
    log_event("condition_start", participant_id=participant_id, position=request.position,
              code=code, label=CONDITIONS[code]["label"])
    return {"response": GREETING, "audio": speak(code, GREETING),
            "task": tasks[(request.position - 1) % len(tasks)] if tasks else ""}


@app.post("/api/message")
def message(request: MessageRequest):
    participant_id, code = condition_for(request.token, request.position)

    with dialogs_lock:
        dialog = dialogs.get((participant_id, request.position))
    if dialog is None:
        raise HTTPException(status_code=409, detail="conversation expired")

    utterance = request.text.strip().lower()[:500]
    dialog_act = classify_utterance(utterance)
    state_before = dialog["state"]["state"]
    dialog["state"], response, act_used = web_dialog.transition(dialog["state"], utterance, dialog_act)
    dialog["turns"] += 1

    restaurant = dialog["state"]["restaurant"]
    log_event("turn", participant_id=participant_id, position=request.position, code=code,
              label=CONDITIONS[code]["label"], turn=dialog["turns"], state_before=state_before,
              user_utterance=utterance, dialog_act=dialog_act, act_used=act_used, state_after=dialog["state"]["state"],
              system_response=response, area=dialog["state"]["area"], food=dialog["state"]["food"],
              pricerange=dialog["state"]["pricerange"],
              recommendation=restaurant["restaurantname"] if restaurant else None)

    return {"response": response, "audio": speak(code, response),
            "finished": dialog["state"]["state"] >= 11, "turns": dialog["turns"]}


@app.post("/api/finish")
def finish(request: StartRequest):
    participant_id, code = condition_for(request.token, request.position)

    with dialogs_lock:
        dialog = dialogs.pop((participant_id, request.position), None)

    log_event("condition_end", participant_id=participant_id, position=request.position, code=code,
              label=CONDITIONS[code]["label"], turns=dialog["turns"] if dialog else None)
    return {"form_url": form_link(FORM_URL, participant_id, code, request.position)}


@app.post("/api/complete")
def complete(request: SessionRequest):
    participant_id, _ = read_token(request.token)
    log_event("session_complete", participant_id=participant_id)
    return {"form_url": form_link(FINAL_FORM_URL, participant_id)}


@app.get("/api/audio/{key}.mp3")
def audio(key: str):
    data = speech.get(key)
    if data is None:
        raise HTTPException(status_code=404)
    return Response(content=data, media_type="audio/mpeg",
                    headers={"Cache-Control": "public, max-age=86400"})


app.mount("/static", StaticFiles(directory=WEBAPP / "static"), name="static")
