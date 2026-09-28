"""Backend diagnostics: Hindsight without Groq.

Usage (backend running):
    venv/Scripts/python.exe scripts/test_hindsight.py --base-url http://localhost:8001
"""
import argparse
import json
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    args = ap.parse_args()
    print("== GET /diagnostics/hindsight ==")
    try:
        with urllib.request.urlopen(args.base_url + "/diagnostics/hindsight", timeout=90) as r:
            print(json.dumps(json.load(r), indent=2)[:3000])
    except Exception as e:  # noqa: BLE001
        print(f"FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
