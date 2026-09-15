"""
Flask API behind https://api.nicholasbien.com (successor to
openmusenet2/web_server_modal.py). Deployed on Railway; inference runs on the
`midigenai-serve` Modal app (see modal_serve.py).

Changes from the old server:
- v1 (GPT-2) is retired. The historical v1 routes (`/api/upload_midi`,
  `/api/generate_from_selected/<f>`) keep their paths and response shapes but
  are served by the current model, so the existing frontend works unchanged.
- The v1 text-streaming routes (`/api/generate`, `/api/generate_stream`)
  return 410 Gone: they spoke the v1 text encoding, which no longer exists.
- `/api/upload_midi_ab` now returns two samples from the current model
  (position-randomized); preference logging is unchanged, so the RLHF feed
  keeps flowing with `model=v2,v2` pairs recorded in ab_pairs.csv.

Preference data (the point of the A/B routes) is written twice: the legacy
CSVs, unchanged, and a `site_pairs/` directory in `label_app`'s exact layout
(pairs/<id>_prompt.mid, _a.mid, _b.mid, <id>.json, labels.jsonl) so
`reward_align --labels site_pairs/labels.jsonl` fits site votes with no
special case. A vote is only worth something if it identifies the sample it
was cast on, so the display order is recorded at generation time -- the
historical loss of ~166 clicks was exactly this, votes whose pair could not
be reconstructed.

All of it lands under DATA_DIR, which MUST be a mounted volume in
production: Railway's container filesystem is wiped on every deploy. Set
MIDIGENAI_DATA_DIR to the mount point. `/` reports `storage`, whose `id`
survives a restart if and only if the data is durable -- if that id changes
after a deploy, everything collected since the last one is gone.

Run locally:  python -m midigenai.web_server          (port 5555)
Production:   ENV=prod, and PORT is honored (Railway sets it).
"""

import datetime
import json
import os
import random
import traceback
import uuid

import modal as _modal
from flask import Flask, Response, jsonify, request, send_from_directory, url_for
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

# The version allowlist lives with the Modal app so the two can't drift.
from midigenai.attributes import control_vocab, validate_controls
from midigenai.modal_serve import SERVED_VERSIONS, resolve_version

env = os.environ.get("ENV", "dev")

# Absolute paths: Flask's send_from_directory resolves relative dirs against
# the package dir (app.root_path), not the CWD.
# Where everything worth keeping goes. In production this has to be a
# mounted volume; the default is the old hard-coded path so an existing
# deployment behaves identically until the mount point is set.
DATA_DIR = os.environ.get("MIDIGENAI_DATA_DIR") or (
    "/app/data" if env == "prod" else os.path.abspath("."))
UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploaded_midi")
GENERATED_FOLDER = os.path.join(DATA_DIR, "generated_midi")
PRESELECTED_FOLDER = os.path.abspath("preselected_midi")  # static, shipped with the repo

# label_app's layout, so the two label sources merge without a converter.
PAIRS_ROOT = os.path.join(DATA_DIR, "site_pairs")
PAIRS_FOLDER = os.path.join(PAIRS_ROOT, "pairs")
LABELS_PATH = os.path.join(PAIRS_ROOT, "labels.jsonl")
EVENTS_PATH = os.path.join(PAIRS_ROOT, "events.jsonl")

for folder in (UPLOAD_FOLDER, GENERATED_FOLDER, PAIRS_FOLDER):
    os.makedirs(folder, exist_ok=True)


def _storage_marker() -> dict:
    """Identity of the data directory, written once and read ever after.

    There is no way to ask the process whether its disk is a volume, but
    there is a way to find out: if this id is the same after a deploy, the
    data survived; if it is new, the container filesystem was wiped and so
    was every vote since the last deploy. Reported by `/`.
    """
    path = os.path.join(DATA_DIR, "storage_id.json")
    try:
        if os.path.exists(path):
            marker = json.loads(open(path).read())
            marker["reused"] = True
            return marker
        marker = {"id": uuid.uuid4().hex[:12],
                  "created": datetime.datetime.now(datetime.timezone.utc)
                  .strftime("%Y-%m-%dT%H:%M:%SZ")}
        with open(path, "w") as f:
            json.dump(marker, f)
        marker["reused"] = False
        return marker
    except (OSError, ValueError) as e:
        # unreadable or corrupt: report it, never fail to boot over it
        return {"id": None, "error": str(e)}


STORAGE = _storage_marker()
if not STORAGE.get("reused"):
    print(f"[data] NEW storage id {STORAGE.get('id')} at {DATA_DIR}. If this "
          f"changes on every deploy, DATA_DIR is ephemeral and preference "
          f"data does not survive -- mount a volume and set MIDIGENAI_DATA_DIR.")
else:
    print(f"[data] storage id {STORAGE['id']} at {DATA_DIR} "
          f"(created {STORAGE.get('created')}) -- durable across this restart.")


def _utcnow() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_jsonl(path: str, row: dict) -> None:
    """One JSON object per line, opened per write. Logging must never be the
    reason a generation request fails."""
    try:
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        print(f"[data] could not append to {path}: {e}")


def _write_pair(pair_id: str, prompt_bytes: bytes, result: dict, version: str,
                params: dict, shown: list[str], route: str) -> bool:
    """Write one A/B pair in label_app's layout. Returns whether it landed.

    `shown` is the display order -- shown[0] is the sample the client got as
    option 1 -- and it is written now, with the pair, because by the time the
    vote arrives nothing else remembers which sample was on which side.

    The scored files are the continuations alone (`cont_midis`), matching
    what label_app writes and what reward_align expects; a deployment of the
    Modal app that predates those fields simply logs no pair rather than
    logging a differently-shaped one.
    """
    conts = result.get("cont_midis")
    if not conts or len(conts) < 2:
        _append_jsonl(EVENTS_PATH, {
            "ts": _utcnow(), "kind": "pair_skipped", "route": route,
            "pair_id": pair_id, "model": version,
            "reason": "serving deployment returns no continuation-only MIDI; "
                      "redeploy modal_serve",
        })
        return False
    try:
        with open(os.path.join(PAIRS_FOLDER, f"{pair_id}_prompt.mid"), "wb") as f:
            f.write(prompt_bytes)
        for name, midi in zip(("a", "b"), conts):
            with open(os.path.join(PAIRS_FOLDER, f"{pair_id}_{name}.mid"), "wb") as f:
                f.write(midi)
        cont_ids = result.get("cont_ids") or [[], []]
        meta = {
            "pair_id": pair_id,
            "created": _utcnow(),
            # the seed prompt groups pairs for leave-one-prompt-out CV
            "prompt_file": f"{pair_id}_prompt.mid",
            "prompt_ids": result.get("prompt_ids", []),
            "cont_a_ids": cont_ids[0],
            "cont_b_ids": cont_ids[1],
            "model_a": version,
            "model_b": version,
            "cross_model": False,
            "tempo_bpm": result.get("tempo_bpm"),
            "source": "site",
            "route": route,
            "shown": shown,
            **params,
        }
        with open(os.path.join(PAIRS_FOLDER, f"{pair_id}.json"), "w") as f:
            json.dump(meta, f)
        return True
    except OSError as e:
        print(f"[data] could not write pair {pair_id}: {e}")
        return False


def _read_pair_meta(pair_id: str) -> dict | None:
    try:
        with open(os.path.join(PAIRS_FOLDER, f"{pair_id}.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _record_vote(pair_id: str, option: str, route: str) -> bool:
    """Append a label_app-schema vote row. `option` is the client's
    option1/option2, resolved through the stored display order to the
    canonical a/b that reward_align reads."""
    meta = _read_pair_meta(pair_id)
    if meta is None:
        _append_jsonl(EVENTS_PATH, {
            "ts": _utcnow(), "kind": "vote_orphan", "route": route,
            "pair_id": pair_id, "option": option,
            "reason": "no pair metadata: the generation predates pair logging, "
                      "or the storage it was written to did not survive",
        })
        return False
    shown = meta.get("shown") or ["a", "b"]
    side = "left" if option == "option1" else "right"
    preferred = shown[0] if side == "left" else shown[1]
    _append_jsonl(LABELS_PATH, {
        "ts": _utcnow(),
        "session_id": f"site:{route}",
        "pair_id": pair_id,
        "choice": side,
        "left_is": shown[0],
        "right_is": shown[1],
        "preferred": preferred,
        "left_model": meta.get(f"model_{shown[0]}", ""),
        "right_model": meta.get(f"model_{shown[1]}", ""),
        "flags": {},
    })
    return True

# Inference: looked up by name so the Modal app needn't run locally.
# One handle per served version; each maps to its own Modal container pool.
_MidiGen = _modal.Cls.from_name("midigenai-serve", "MidiGen")
_gen_handles: dict[str, object] = {}


def _generator(version: str):
    """Modal handle for one model version, created once and reused."""
    if version not in _gen_handles:
        _gen_handles[version] = _MidiGen(version=version)
    return _gen_handles[version]

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GENERATED_FOLDER"] = GENERATED_FOLDER
app.config["PRESELECTED_FOLDER"] = PRESELECTED_FOLDER

CORS(app)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_port=1, x_prefix=1)


def _unique_string() -> str:
    return f"{datetime.datetime.now():%Y%m%d%H%M%S}_{uuid.uuid4().hex}"


def _generate_n(midi_bytes: bytes, temperature: float, top_k: int,
                max_new_tokens: int, n_samples: int, version: str,
                controls: dict | None = None) -> dict:
    """n continuations in ONE Modal call, batched on the GPU — one round trip
    and a shared prefill instead of n sequential generations.

    `controls` is only sent when the caller asked for one: a serving
    deployment older than the control surface rejects the argument, and an
    uncontrolled request should not depend on the deploy order of two
    services."""
    kwargs = {"controls": controls} if controls else {}
    result = _generator(version).generate_batch.remote(
        midi_bytes, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, n_samples=n_samples, **kwargs,
    )
    return result


def _prompt_fields(result: dict) -> dict:
    """Where the model's input ended, for the client's prompt display. A
    prompt longer than the context window is cut at the end (see
    generate.fit_to_context); the continuation follows the cut."""
    return {
        "promptTruncated": bool(result.get("prompt_truncated", False)),
        "promptEndSeconds": result.get("prompt_end_seconds"),
        "promptTokens": result.get("prompt_tokens"),
        "promptTokensTotal": result.get("prompt_tokens_total"),
    }


def _save_midi(midi_bytes: bytes, base_name: str, suffix: str,
               version: str = "model") -> str:
    filename = f"{version}_{base_name}_{suffix}.mid"
    out_path = os.path.join(GENERATED_FOLDER, filename)
    with open(out_path, "wb") as f:
        f.write(midi_bytes)
    return out_path


def _read_upload():
    """Validate the multipart upload; returns (midi_bytes, base_name) or a Response."""
    if "midiFile" not in request.files:
        return Response("No file part", status=400)
    file = request.files["midiFile"]
    if file.filename == "":
        return Response("No selected file", status=400)
    file.seek(0, os.SEEK_END)
    if file.tell() > 1 * 1024 * 1024:
        return Response("File is too large", status=400)
    file.seek(0)
    return file.read(), os.path.splitext(secure_filename(file.filename))[0]


def _two_samples_response(midi_bytes: bytes, base: str, temperature: float,
                          top_k: int, max_new_tokens: int, version: str,
                          route: str = "generate", controls: dict | None = None):
    unique_str = _unique_string()
    result = _generate_n(midi_bytes, temperature, top_k, max_new_tokens,
                         n_samples=2, version=version, controls=controls)
    out_paths = [
        _save_midi(m, f"{base}_{unique_str}", str(i), version)
        for i, m in enumerate(result["midis"])
    ]
    # Option 1 is sample a here (no shuffle on this route), recorded anyway so
    # the vote path never has to assume it.
    _write_pair(unique_str, midi_bytes, result, version,
                {"temperature": temperature, "top_k": top_k,
                 "max_new_tokens": max_new_tokens, "controls": controls or {},
                 "header": result.get("header", [])},
                shown=["a", "b"], route=route)
    return jsonify({
        "message": "MIDI file generated successfully",
        "model": version,
        # Echoing this back is what lets a vote name the samples it was cast
        # on; the frontend can send it to /api/submit_preference as
        # `requestId` (older clients that don't are still accepted).
        "requestId": unique_str,
        "midiUrl1": url_for("serve_user_midi",
                            filename=os.path.basename(out_paths[0]), _external=True),
        "midiUrl2": url_for("serve_user_midi",
                            filename=os.path.basename(out_paths[1]), _external=True),
        # what the model was actually told, controls included
        "header": result.get("header", []),
        **_prompt_fields(result),
    })


def _gen_params():
    return (
        request.args.get("temperature", default=1.2, type=float),
        request.args.get("top_k", default=50, type=int),
        request.args.get("max_new_tokens", default=512, type=int),
    )


def _controls() -> dict:
    """Attribute-header controls from the query string or form.

    `density`, `poly` and `range` are the buckets the dataset builder
    computed from the notes (see /, which publishes the vocabulary);
    `instruments` is a comma-separated list of families, `genre` one name.
    Validated here so a typo is a 400 naming the allowed values rather than
    a 500 from a vocabulary lookup three processes away. Returns {} when the
    caller asked for nothing, which keeps those requests on the code path
    they have always taken.
    """
    def _get(name):
        return request.args.get(name) or request.form.get(name)

    raw = {"density": _get("density"), "poly": _get("poly"),
           "pitch_range": _get("range")}
    instruments = _get("instruments")
    if instruments:
        raw["instruments"] = [i.strip() for i in instruments.split(",") if i.strip()]
    genre = _get("genre")
    if genre:
        raw["genres"] = [genre.strip()]
    asked = {k: v for k, v in raw.items() if v not in (None, "", [])}
    return validate_controls(**asked) if asked else {}


def _model_version() -> str:
    """The checkpoint the request asked for.

    The site's dropdown sends `model=v2|v3|v4`; anything unrecognized falls
    back to the default rather than erroring, so an old cached frontend keeps
    working. The returned value is a volume subfolder, not raw user input.
    """
    asked = request.args.get("model") or request.form.get("model")
    return resolve_version(asked)


@app.route("/")
def health():
    def _count(path):
        try:
            with open(path) as f:
                return sum(1 for line in f if line.strip())
        except OSError:
            return 0

    return jsonify({
        "service": "midigenai api",
        "models": sorted(SERVED_VERSIONS),
        "default_model": resolve_version(None),
        # If `id` changes after a deploy, DATA_DIR is not a volume and the
        # counts below went with it.
        # what /api/* will accept as controls, so a client needn't hardcode it
        "controls": control_vocab(),
        "storage": {
            "dir": DATA_DIR,
            "id": STORAGE.get("id"),
            "created": STORAGE.get("created"),
            "pairs": len([f for f in os.listdir(PAIRS_FOLDER)
                          if f.endswith(".json")]) if os.path.isdir(PAIRS_FOLDER) else 0,
            "labels": _count(LABELS_PATH),
        },
    })


# ---------- generation ---------- #

# The old v1 route and the _v2 route now share one implementation; both
# response shapes were already identical.
@app.route("/api/upload_midi", methods=["POST"])
@app.route("/api/upload_midi_v2", methods=["POST"])
def upload_midi():
    temperature, top_k, max_new_tokens = _gen_params()
    version = _model_version()
    upload = _read_upload()
    if isinstance(upload, Response):
        return upload
    midi_bytes, base = upload
    try:
        controls = _controls()
    except ValueError as e:
        return jsonify({"error": str(e), "controls": control_vocab()}), 400
    try:
        return _two_samples_response(midi_bytes, base, temperature, top_k,
                                     max_new_tokens, version, route="upload",
                                     controls=controls)
    except ValueError as e:
        # the model can't do what was asked (controls on v3, for instance)
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate_from_selected/<filename>")
@app.route("/api/generate_from_selected_v2/<filename>")
def generate_from_selected(filename):
    temperature, top_k, max_new_tokens = _gen_params()
    version = _model_version()
    filename = secure_filename(filename)
    input_filepath = os.path.join(app.config["PRESELECTED_FOLDER"], filename)
    if not os.path.exists(input_filepath):
        return jsonify({"error": "File not found"}), 404
    try:
        controls = _controls()
    except ValueError as e:
        return jsonify({"error": str(e), "controls": control_vocab()}), 400
    try:
        with open(input_filepath, "rb") as fh:
            midi_bytes = fh.read()
        base = os.path.splitext(filename)[0]
        return _two_samples_response(midi_bytes, base, temperature, top_k,
                                     max_new_tokens, version, route="preselected",
                                     controls=controls)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload_midi_ab", methods=["POST"])
def upload_midi_ab():
    """Two samples from the current model, position-randomized; the pair is
    logged so preferences remain usable for reward-model training."""
    temperature, top_k, max_new_tokens = _gen_params()
    version = _model_version()
    upload = _read_upload()
    if isinstance(upload, Response):
        return upload
    midi_bytes, base = upload
    unique_str = _unique_string()

    input_path = os.path.join(UPLOAD_FOLDER, f"ab_{base}_{unique_str}.mid")
    with open(input_path, "wb") as f:
        f.write(midi_bytes)

    try:
        controls = _controls()
    except ValueError as e:
        return jsonify({"error": str(e), "controls": control_vocab()}), 400
    try:
        result = _generate_n(midi_bytes, temperature, top_k, max_new_tokens,
                             n_samples=2, version=version, controls=controls)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    paths = [
        _save_midi(m, f"{base}_{unique_str}", f"ab{i}", version)
        for i, m in enumerate(result["midis"])
    ]
    # Randomize which sample is option 1, then write that order down. The
    # shuffle without the record is what made the old click data unusable:
    # "option1 won" says nothing once you can't tell which sample it was.
    shown = ["a", "b"]
    if random.random() < 0.5:
        paths.reverse()
        shown.reverse()

    _write_pair(unique_str, midi_bytes, result, version,
                {"temperature": temperature, "top_k": top_k,
                 "max_new_tokens": max_new_tokens, "controls": controls,
                 "header": result.get("header", [])},
                shown=shown, route="ab")

    with open(os.path.join(DATA_DIR, "ab_pairs.csv"), "a") as f:
        f.write(f"{unique_str},{base},{version},{version}\n")

    return jsonify({
        "message": "A/B MIDI generated",
        "requestId": unique_str,
        "model": version,
        "midiUrl1": url_for("serve_user_midi",
                            filename=os.path.basename(paths[0]), _external=True),
        "midiUrl2": url_for("serve_user_midi",
                            filename=os.path.basename(paths[1]), _external=True),
        "header": result.get("header", []),
        **_prompt_fields(result),
    })


@app.route("/api/accompany", methods=["POST"])
def accompany():
    """Parts to play WITH the upload, not after it.

    Continuation answers "what happens next"; this answers "what else is
    playing". The model writes over the same bars the upload occupies, and
    each returned file is the upload's chosen track with one generated
    answer stacked on it, so it plays as a duet rather than a handoff.

    v4 only: earlier checkpoints have no accompaniment document type.
    """
    temperature, top_k, _ = _gen_params()
    version = _model_version()
    bars = request.args.get("bars", default=8, type=int)
    bars = max(1, min(bars, 32))
    upload = _read_upload()
    if isinstance(upload, Response):
        return upload
    midi_bytes, base = upload
    unique_str = _unique_string()

    try:
        controls = _controls()
    except ValueError as e:
        return jsonify({"error": str(e), "controls": control_vocab()}), 400
    try:
        result = _generator(version).accompany_batch.remote(
            midi_bytes, bars=bars, temperature=temperature, top_k=top_k,
            n_samples=2, **({"controls": controls} if controls else {}),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    paths = [
        _save_midi(m, f"{base}_{unique_str}", f"acc{i}", version)
        for i, m in enumerate(result["midis"])
    ]
    return jsonify({
        "message": "Accompaniment generated",
        "requestId": unique_str,
        "model": version,
        "midiUrl1": url_for("serve_user_midi",
                            filename=os.path.basename(paths[0]), _external=True),
        "midiUrl2": url_for("serve_user_midi",
                            filename=os.path.basename(paths[1]), _external=True),
        # The whole returned file is condition + answer playing together, so
        # there is no prompt/continuation cut for the client to mark.
        "bars": result["bars"],
        "barsAvailable": result["bars_available"],
        "windowSeconds": result["window_seconds"],
        "conditionTrack": result["condition_track"],
        "conditionTrackName": result["condition_track_name"],
        "trackNames": result["track_names"],
        "generatedNotes": result["generated_notes"],
        "header": result.get("header", []),
    })


@app.route("/api/infill", methods=["POST"])
def infill():
    """Rewrite bars [at_bar, at_bar+bars) of the upload, in place.

    Continuation answers "what happens next" and accompaniment answers "what
    else is playing"; this is "that bit in the middle, again". The model sees
    the music on both sides of the gap, so the new span has to land back on
    what follows it. Each returned file is the whole piece with the span
    replaced, so the client plays it against the original and hears one
    section change.

    v4 only: earlier checkpoints have no span-infill document type.
    """
    temperature, top_k, _ = _gen_params()
    version = _model_version()
    at_bar = max(0, request.args.get("at_bar", default=0, type=int))
    bars = max(1, min(request.args.get("bars", default=2, type=int), 32))
    upload = _read_upload()
    if isinstance(upload, Response):
        return upload
    midi_bytes, base = upload
    unique_str = _unique_string()

    try:
        controls = _controls()
    except ValueError as e:
        return jsonify({"error": str(e), "controls": control_vocab()}), 400
    try:
        result = _generator(version).infill_batch.remote(
            midi_bytes, at_bar=at_bar, bars=bars, temperature=temperature,
            top_k=top_k, n_samples=2,
            **({"controls": controls} if controls else {}),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

    paths = [
        _save_midi(m, f"{base}_{unique_str}", f"fill{i}", version)
        for i, m in enumerate(result["midis"])
    ]
    return jsonify({
        "message": "Infill generated",
        "requestId": unique_str,
        "model": version,
        "midiUrl1": url_for("serve_user_midi",
                            filename=os.path.basename(paths[0]), _external=True),
        "midiUrl2": url_for("serve_user_midi",
                            filename=os.path.basename(paths[1]), _external=True),
        # where the rewritten span sits in the returned files
        "atBar": result["at_bar"],
        "bars": result["bars"],
        "barsAvailable": result["bars_available"],
        "spanStartSeconds": result["span_start_seconds"],
        "spanSeconds": result["span_seconds"],
        "generatedNotes": result["generated_notes"],
        "header": result.get("header", []),
    })



# v1 text-encoding streaming routes: the text format is retired.
@app.route("/api/generate", methods=["POST"])
@app.route("/api/generate_stream", methods=["POST"])
def generate_text_retired():
    return jsonify({
        "error": "This endpoint served the retired v1 text-encoding model. "
                 "Use /api/upload_midi instead.",
    }), 410


# ---------- files & preferences ---------- #

@app.route("/file/preselected_midi/<filename>")
def serve_preselected_midi(filename):
    return send_from_directory(app.config["PRESELECTED_FOLDER"], filename)


@app.route("/file/user_midi/<filename>")
def serve_user_midi(filename):
    return send_from_directory(app.config["GENERATED_FOLDER"], filename)


@app.route("/api/list_midi")
def list_midi():
    files = os.listdir(app.config["PRESELECTED_FOLDER"])
    return jsonify([f for f in files if f.endswith((".mid", ".midi"))])


@app.route("/api/submit_preference", methods=["POST"])
def record_preference():
    preferred_option = request.form.get("preferredMidi")
    if not preferred_option:
        return jsonify({"error": "No preference provided"}), 400
    file_name = request.form.get("selectedFileName", "")
    input_midi_name = os.path.splitext(os.path.basename(file_name))[0]
    response_value = {"option1": 0, "option2": 1}.get(preferred_option, -1)
    with open(os.path.join(DATA_DIR, "responses.csv"), "a") as f:
        f.write(f"{_unique_string()},{input_midi_name},{response_value}\n")
    # A frontend that echoes `requestId` back gets a real label row; one that
    # doesn't still gets the CSV, and the JSONL says why the vote is orphaned
    # rather than leaving a silent gap.
    paired = _record_vote(request.form.get("requestId", ""), preferred_option,
                          route="generate")
    return jsonify({"message": "Preference recorded successfully",
                    "paired": paired})


@app.route("/api/submit_preference_ab", methods=["POST"])
def record_preference_ab():
    request_id = request.form.get("requestId")
    preferred_option = request.form.get("preferredMidi")
    if not request_id or not preferred_option:
        return jsonify({"error": "requestId and preferredMidi required"}), 400
    response_value = 0 if preferred_option == "option1" else 1
    with open(os.path.join(DATA_DIR, "responses_ab.csv"), "a") as f:
        f.write(f"{request_id},{response_value}\n")
    paired = _record_vote(request_id, preferred_option, route="ab")
    return jsonify({"message": "A/B preference recorded", "paired": paired})


def main():
    port = int(os.environ.get("PORT", "5000" if env == "prod" else "5555"))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
