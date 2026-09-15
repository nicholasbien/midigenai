# Preference data from the public site

The demo at nicholasbien.com/midi generates two continuations per upload and
asks which one is better. That is the same question `label_app` asks, from
people who are not the author, on music they brought themselves — the most
on-distribution preference data available. Historically almost none of it
survived: ~166 clicks were lost because the server never recorded which
sample was on which side, and everything written since has been going to a
Railway container filesystem that is wiped on every deploy.

Both halves are fixed in `web_server.py`. This is the operator's side of it.

## 1. Attach a volume, or nothing here matters

Railway volumes are configured per service in the dashboard (Service →
Settings → Volumes), not in `railway.json`. Mount one, then point the server
at the mount:

    MIDIGENAI_DATA_DIR=/data        # whatever the mount path is

Without the variable the server keeps its old default (`/app/data` in prod),
which is container-local: correct behaviour for an unchanged deployment, and
exactly the thing that loses the data.

**Verify it, don't assume it.** `GET /` reports:

```json
"storage": {"dir": "/data", "id": "9f2c1a7b4e05", "created": "...",
            "pairs": 412, "labels": 388}
```

`id` is written once into the data directory and read ever after. Deploy
again and re-check: **same id, the disk is durable; new id, it is not** and
everything collected since the last deploy is gone. The startup log says
which case it saw.

## 2. What gets written

Under `$MIDIGENAI_DATA_DIR/site_pairs/`, in `label_app`'s exact layout:

```
labels.jsonl              one line per vote (label_app schema)
events.jsonl              orphaned votes, skipped pairs — the gaps, named
pairs/<id>.json           model, sampling params, token ids, display order
pairs/<id>_prompt.mid     what the visitor uploaded
pairs/<id>_a.mid          continuation A, alone and re-zeroed
pairs/<id>_b.mid          continuation B
```

`_a.mid` / `_b.mid` are the continuations *only*, matching what `label_app`
writes — the reward is fitted on the continuation, not on the prompt the two
sides share. The Modal serving app returns them (`cont_midis`), so
**redeploy `modal_serve` before expecting pairs**: an older deployment
returns no continuation-only MIDI and the server logs a `pair_skipped` event
rather than writing a differently-shaped pair.

`shown` in the pair JSON is the display order at generation time. A/B still
randomizes which sample is option 1; the difference is that the order is now
written down before the vote arrives, which is the whole reason the old
clicks were unusable.

## 3. Fit it

The layout is the one `reward_align` already reads, so site votes and
labeling-app votes merge with no converter:

    modal volume get ...          # or: railway ssh / volume download
    python -m midigenai.reward_align --labels site_pairs/labels.jsonl \
        --out evals/reward/reward_site.json

Cross-validation groups by `prompt_file`, which for site pairs is the
visitor's own upload, so leave-one-prompt-out still means what it says.

## 4. One frontend change, worth making

Generation responses now include `requestId`. The A/B route already round-
trips it; the plain route's vote (`/api/submit_preference`) does not, so
those votes land in `responses.csv` and in `events.jsonl` as `vote_orphan`
until the site echoes it back:

    fetch("/api/submit_preference", {method: "POST", body: form})
    // form.append("requestId", data.requestId)   <- this line

Until then the A/B route is the one producing fittable pairs.
