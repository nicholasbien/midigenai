"""
Tiny wrapper around the Lambda Cloud API.

Usage:
    export LAMBDA_API_KEY=lambdaapi-...
    python -m midigenai.lambda_provision list-types
    python -m midigenai.lambda_provision launch --instance-type gpu_1x_h100_pcie --ssh-key mykey
    python -m midigenai.lambda_provision wait <instance-id>
    python -m midigenai.lambda_provision list
    python -m midigenai.lambda_provision terminate <instance-id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

import urllib.request
import urllib.error
import base64


API_BASE = "https://cloud.lambdalabs.com/api/v1"


def _auth_header(api_key: str) -> dict[str, str]:
    # Lambda uses HTTP Basic auth with api_key as username and empty password.
    token = base64.b64encode(f"{api_key}:".encode()).decode()
    return {"Authorization": f"Basic {token}"}


def _request(method: str, path: str, api_key: str, body: dict | None = None) -> Any:
    url = f"{API_BASE}{path}"
    headers = {
        "Content-Type": "application/json",
        # Cloudflare blocks the default Python-urllib user-agent — masquerade as a normal client.
        "User-Agent": "midigenai/0.1 (curl-equivalent)",
        "Accept": "application/json",
        **_auth_header(api_key),
    }
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code} on {method} {path}: {body}") from None


def get_api_key() -> str:
    key = os.environ.get("LAMBDA_API_KEY")
    if key:
        return key
    env_file = os.path.expanduser("~/.lambda_env")
    if os.path.exists(env_file):
        for line in open(env_file):
            line = line.strip()
            if line.startswith("LAMBDA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("LAMBDA_API_KEY not set; export it or put LAMBDA_API_KEY=... in ~/.lambda_env")


# ---------------- commands ---------------- #

def list_types(args):
    api_key = get_api_key()
    data = _request("GET", "/instance-types", api_key)["data"]
    rows = []
    for name, info in data.items():
        regions = info.get("regions_with_capacity_available", [])
        if not regions:
            continue
        spec = info["instance_type"]
        rows.append({
            "name": name,
            "gpu_description": spec.get("gpu_description", ""),
            "price_cents_per_hour": spec.get("price_cents_per_hour", 0),
            "regions": [r["name"] for r in regions],
        })
    rows.sort(key=lambda r: r["price_cents_per_hour"])
    print(json.dumps(rows, indent=2))


def list_instances(args):
    api_key = get_api_key()
    data = _request("GET", "/instances", api_key)["data"]
    print(json.dumps(data, indent=2))


def list_keys(args):
    api_key = get_api_key()
    data = _request("GET", "/ssh-keys", api_key)["data"]
    for k in data:
        pub = k.get("public_key", "")
        # show the key fingerprint (last 30 chars of base64) for matching
        tail = pub.split()[1][-30:] if " " in pub else pub[-30:]
        print(f"{k['name']:30s}  ...{tail}  {k.get('id','')}")


def launch(args):
    api_key = get_api_key()
    body = {
        "region_name": args.region,
        "instance_type_name": args.instance_type,
        "ssh_key_names": [args.ssh_key],
        "file_system_names": [],
        "quantity": 1,
        "name": args.name,
    }
    print(f"[launch] {body}")
    resp = _request("POST", "/instance-operations/launch", api_key, body)
    instance_ids = resp["data"]["instance_ids"]
    print(f"[launch] instance_ids={instance_ids}")
    for iid in instance_ids:
        print(iid)


def wait(args):
    api_key = get_api_key()
    iid = args.instance_id
    print(f"[wait] polling {iid}...")
    while True:
        data = _request("GET", f"/instances/{iid}", api_key)["data"]
        status = data.get("status")
        ip = data.get("ip")
        print(f"[wait] status={status} ip={ip}")
        if status == "active" and ip:
            print(json.dumps(data, indent=2))
            return
        if status in ("terminated", "failed"):
            sys.exit(f"instance {iid} {status}")
        time.sleep(15)


def terminate(args):
    api_key = get_api_key()
    body = {"instance_ids": [args.instance_id]}
    resp = _request("POST", "/instance-operations/terminate", api_key, body)
    print(json.dumps(resp, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-types").set_defaults(func=list_types)
    sub.add_parser("list").set_defaults(func=list_instances)
    sub.add_parser("list-keys").set_defaults(func=list_keys)

    pl = sub.add_parser("launch")
    pl.add_argument("--instance-type", required=True,
                    help="e.g. gpu_1x_h100_pcie or gpu_1x_a100")
    pl.add_argument("--region", required=True, help="e.g. us-west-1")
    pl.add_argument("--ssh-key", required=True, help="name of an SSH key in your account")
    pl.add_argument("--name", default="midigenai", help="instance name tag")
    pl.set_defaults(func=launch)

    pw = sub.add_parser("wait")
    pw.add_argument("instance_id")
    pw.set_defaults(func=wait)

    pt = sub.add_parser("terminate")
    pt.add_argument("instance_id")
    pt.set_defaults(func=terminate)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
