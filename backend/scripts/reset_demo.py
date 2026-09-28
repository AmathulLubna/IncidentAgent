"""Reset demo SQLite state (Hindsight banks untouched).

Usage:
    venv/Scripts/python.exe scripts/reset_demo.py --base-url http://localhost:8000
"""
import argparse
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    args = ap.parse_args()
    payload = json.dumps({"confirm": True}).encode()
    req = urllib.request.Request(args.base_url + "/demo/reset", data=payload,
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        print(json.dumps(json.load(r), indent=2))


if __name__ == "__main__":
    main()
