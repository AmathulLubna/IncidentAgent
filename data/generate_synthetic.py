"""
Generates synthetic incidents across 5 recurring root-cause families using
Groq. Falls back to hand-written templates (varied slightly) if the API is
unavailable, so this script is safe to run even before you've added keys.

Usage:
    python data/generate_synthetic.py --per-family 4 --out data/synthetic_incidents.json
"""
import argparse
import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

FAMILIES = {
    "db_connection_pool_exhaustion": {
        "services": ["orders-api", "billing-service", "checkout-api"],
        "seed_symptom": "Requests timing out with 'connection pool exhausted' / 'QueuePool limit reached' errors under load.",
    },
    "auth_token_expiry": {
        "services": ["auth-service", "gateway-api", "session-service"],
        "seed_symptom": "Users getting spurious 401s; logs show token-refresh calls failing right after a deploy touching the auth module.",
    },
    "memory_leak": {
        "services": ["notification-worker", "image-processor", "search-indexer"],
        "seed_symptom": "Pod memory climbs steadily until OOMKilled; restarts temporarily fix it, recurs every few hours.",
    },
    "rate_limit_breach": {
        "services": ["payments-api", "webhook-dispatcher", "public-api"],
        "seed_symptom": "Downstream 429s from a third-party API cascading into our own request queue backing up.",
    },
    "bad_deploy_config": {
        "services": ["recommendation-service", "search-api", "inventory-service"],
        "seed_symptom": "Error rate spikes immediately after a deploy; rollback resolves it. Config diff shows an env var change.",
    },
}


def _try_groq_generate(family: str, seed_symptom: str, services: list[str], n: int) -> list[dict] | None:
    try:
        from app.config import settings
        from openai import OpenAI

        if not settings.GROQ_API_KEY:
            return None
        client = OpenAI(api_key=settings.GROQ_API_KEY, base_url=settings.GROQ_BASE_URL)
        prompt = (
            f"Generate {n} realistic but DIFFERENT production-incident symptom descriptions "
            f"(2-3 sentences each, include a plausible log line) for the root cause family "
            f"'{family}'. Seed example: {seed_symptom}\n"
            f"Vary the specific service (pick from: {', '.join(services)}), the numbers, and the wording.\n"
            "Return ONLY a JSON array of strings, nothing else."
        )
        resp = client.chat.completions.create(
            model=settings.GROQ_MODEL_PRIMARY,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
        )
        text = resp.choices[0].message.content.strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(text)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] Groq generation failed for {family}: {e}. Using template fallback.")
        return None


def _template_fallback(seed_symptom: str, n: int) -> list[str]:
    variants = [
        seed_symptom,
        seed_symptom.replace("timing out", "erroring"),
        seed_symptom + " Started roughly 20 minutes after the last deploy.",
        seed_symptom + " Affects ~15% of traffic during peak hours.",
    ]
    while len(variants) < n:
        variants.append(seed_symptom + f" (variant {len(variants)})")
    return variants[:n]


def generate(per_family: int, seed: int = 42) -> list[dict]:
    random.seed(seed)
    incidents = []
    base_time = datetime.now(timezone.utc) - timedelta(days=30)

    for family, cfg in FAMILIES.items():
        symptoms = _try_groq_generate(family, cfg["seed_symptom"], cfg["services"], per_family)
        if not symptoms:
            symptoms = _template_fallback(cfg["seed_symptom"], per_family)

        for i, symptom in enumerate(symptoms):
            incidents.append(
                {
                    "service": random.choice(cfg["services"]),
                    "symptom": symptom,
                    "timestamp": (base_time + timedelta(hours=random.randint(0, 24 * 30))).isoformat(),
                    "known_root_cause_family": family,
                    "known_fix_steps": _default_fix_steps(family),
                }
            )
    random.shuffle(incidents)
    return incidents


def _default_fix_steps(family: str) -> list[str]:
    return {
        "db_connection_pool_exhaustion": [
            "Check current pool utilization via the DB dashboard",
            "Temporarily raise max pool size / add read replica capacity",
            "Identify and kill any long-running queries holding connections",
            "Add connection timeout + retry-with-backoff on the client",
        ],
        "auth_token_expiry": [
            "Check recent deploys touching the auth/token-refresh module",
            "Roll back the offending deploy if confirmed",
            "Verify token TTL config matches the intended value",
            "Force-refresh cached tokens for affected sessions",
        ],
        "memory_leak": [
            "Capture a heap snapshot before restarting the pod",
            "Restart affected pods to restore service immediately",
            "Diff recent code changes for unbounded caches/listeners",
            "Ship a patch and monitor memory growth over the next cycle",
        ],
        "rate_limit_breach": [
            "Identify which downstream API is rate-limiting us",
            "Add/verify exponential backoff and jitter on retries",
            "Add a local queue + concurrency cap in front of the call",
            "Request a rate-limit increase from the vendor if recurring",
        ],
        "bad_deploy_config": [
            "Diff the config/env vars against the last known-good deploy",
            "Roll back to the previous deploy immediately",
            "Fix the offending config value in a follow-up PR",
            "Add a config-diff check to the deploy pipeline",
        ],
    }.get(family, ["Investigate and document findings", "Apply fix", "Verify resolution"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--out", type=str, default=os.path.join(os.path.dirname(__file__), "synthetic_incidents.json"))
    args = parser.parse_args()

    data = generate(args.per_family)
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {len(data)} synthetic incidents to {args.out}")
