"""Backend diagnostics: Groq without the agent.

Usage (backend running on :8001):
    venv/Scripts/python.exe scripts/test_groq.py --base-url http://localhost:8001
"""
import argparse
import json
import urllib.request


def get(base, path):
    with urllib.request.urlopen(base + path, timeout=90) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    args = ap.parse_args()
    for path in ("/diagnostics/groq", "/diagnostics/groq-tools"):
        print(f"== GET {path} ==")
        try:
            print(json.dumps(get(args.base_url, path), indent=2)[:2000])
        except Exception as e:  # noqa: BLE001
            print(f"FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
