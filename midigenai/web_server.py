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
from midigenai.modal_serve import SERVED_VERSIONS, resolve_version

env = os.environ.get("ENV", "dev")

# Absolute paths: Flask's send_from_directory resolves relative dirs against
# the package dir (app.root_path), not the CWD.
DATA_DIR = "/app/data" if env == "prod" else os.path.abspath(".")
UPLOAD_FOLDER = os.path.join(DATA_DIR, "uploaded_midi")
GENERATED_FOLDER = os.path.join(DATA_DIR, "generated_midi")
PRESELECTED_FOLDER = os.path.abspath("preselected_midi")  # static, shipped with the repo

for folder in (UPLOAD_FOLDER, GENERATED_FOLDER):
    os.makedirs(folder, exist_ok=True)

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
                max_new_tokens: int, n_samples: int, version: str) -> dict:
    """n continuations in ONE Modal call, batched on the GPU — one round trip
    and a shared prefill instead of n sequential generations."""
    result = _generator(version).generate_batch.remote(
        midi_bytes, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, n_samples=n_samples,
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
                          top_k: int, max_new_tokens: int, version: str):
    unique_str = _unique_string()
    result = _generate_n(midi_bytes, temperature, top_k, max_new_tokens,
                         n_samples=2, version=version)
    out_paths = [
        _save_midi(m, f"{base}_{unique_str}", str(i), version)
        for i, m in enumerate(result["midis"])
    ]
    return jsonify({
        "message": "MIDI file generated successfully",
        "model": version,
        "midiUrl1": url_for("serve_user_midi",
                            filename=os.path.basename(out_paths[0]), _external=True),
        "midiUrl2": url_for("serve_user_midi",
                            filename=os.path.basename(out_paths[1]), _external=True),
        **_prompt_fields(result),
    })


def _gen_params():
    return (
        request.args.get("temperature", default=1.2, type=float),
        request.args.get("top_k", default=50, type=int),
        request.args.get("max_new_tokens", default=512, type=int),
    )


ACCOMPANY_INSTRUMENTS = ("Piano", "ChromPerc", "Organ", "Guitar", "Bass", "Strings",
                         "Ensemble", "Brass", "Reed", "Pipe", "SynthLead", "SynthPad",
                         "SynthFX", "Ethnic", "Percussive", "SoundFX", "Drums")


def parse_instruments(raw: str | None) -> list[str]:
    """`instruments=Bass,Drums` -> ["Bass", "Drums"]. Case-insensitive, unknown
    names dropped, order kept, no duplicates. Empty means "no request"."""
    out, lut = [], {n.lower(): n for n in ACCOMPANY_INSTRUMENTS}
    for part in (raw or "").split(","):
        name = lut.get(part.strip().lower())
        if name and name not in out:
            out.append(name)
    return out


def _instruments_param() -> list[str]:
    return parse_instruments(request.args.get("instruments") or request.form.get("instruments"))


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
    return jsonify({
        "service": "midigenai api",
        "models": sorted(SERVED_VERSIONS),
        "default_model": resolve_version(None),
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
        return _two_samples_response(midi_bytes, base, temperature, top_k,
                                     max_new_tokens, version)
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
        with open(input_filepath, "rb") as fh:
            midi_bytes = fh.read()
        base = os.path.splitext(filename)[0]
        return _two_samples_response(midi_bytes, base, temperature, top_k,
                                     max_new_tokens, version)
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

    result = _generate_n(midi_bytes, temperature, top_k, max_new_tokens,
                         n_samples=2, version=version)
    paths = [
        _save_midi(m, f"{base}_{unique_str}", f"ab{i}", version)
        for i, m in enumerate(result["midis"])
    ]
    if random.random() < 0.5:
        paths.reverse()

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
    instruments = _instruments_param()
    upload = _read_upload()
    if isinstance(upload, Response):
        return upload
    midi_bytes, base = upload
    unique_str = _unique_string()

    try:
        result = _generator(version).accompany_batch.remote(
            midi_bytes, bars=bars, temperature=temperature, top_k=top_k,
            n_samples=2, instruments=instruments,
        )
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
        "conditionInstruments": result.get("condition_instruments", []),
        "instrumentsRequested": result.get("instruments_requested", []),
        "trackNames": result["track_names"],
        "generatedNotes": result["generated_notes"],
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
    return jsonify({"message": "Preference recorded successfully"})


@app.route("/api/submit_preference_ab", methods=["POST"])
def record_preference_ab():
    request_id = request.form.get("requestId")
    preferred_option = request.form.get("preferredMidi")
    if not request_id or not preferred_option:
        return jsonify({"error": "requestId and preferredMidi required"}), 400
    response_value = 0 if preferred_option == "option1" else 1
    with open(os.path.join(DATA_DIR, "responses_ab.csv"), "a") as f:
        f.write(f"{request_id},{response_value}\n")
    return jsonify({"message": "A/B preference recorded"})


def main():
    port = int(os.environ.get("PORT", "5000" if env == "prod" else "5555"))
    app.run(host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
