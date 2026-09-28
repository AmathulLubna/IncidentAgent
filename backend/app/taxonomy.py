"""
Incident taxonomy — deterministic understanding of WHAT KIND of incident this is.

Why this module exists
----------------------
Hindsight recall is a fuzzy semantic search: on a small bank it happily returns
the same few memories for *any* query, which made every diagnosis show the same
"historical evidence". Provenance checks (is it resolved? is it the current
incident?) cannot fix that — a resolved disk-full incident is perfectly
"historical" but is not evidence for a 401 storm.

This module gives the application its own, explainable notion of relevance:

* ``classify()``           — rank root-cause categories from the incident text
* ``extract_signals()``    — pull concrete evidence (HTTP codes, %, error names)
* ``similarity()``         — token overlap between two incident texts
* ``topical_verdict()``    — the gate: may this memory count as evidence for
                             THIS incident? (same category, or textually close)

Everything here is pure, offline and unit-testable. No LLM involved.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Category definitions: slug -> (label, [(regex, weight), ...])
# Weight 3 = strong/unique indicator, 2 = good indicator, 1 = weak/contextual.
# ---------------------------------------------------------------------------

_P = re.compile  # shorthand
_I = re.I

CATEGORY_LABELS: dict[str, str] = {
    "db_connection_pool_exhaustion": "Database connection pool exhaustion",
    "auth_token_expiry": "Authentication / token failure",
    "memory_leak": "Memory leak / OOM",
    "rate_limit_breach": "Rate limiting / throttling (429)",
    "bad_deploy_config": "Bad deploy / configuration change",
    "disk_full": "Disk / storage exhaustion",
    "cpu_saturation": "CPU saturation",
    "slow_query_db_lock": "Slow queries / database locking",
    "cache_failure": "Cache failure / stampede",
    "queue_backlog": "Queue / consumer backlog",
    "tls_certificate_expiry": "TLS / certificate failure",
    "dns_network_failure": "DNS / network connectivity failure",
    "dependency_outage": "Upstream / third-party dependency outage",
    "k8s_pod_crashloop": "Kubernetes pod crash / scheduling failure",
    "capacity_saturation": "Traffic surge / capacity saturation",
}

_PATTERNS: dict[str, list[tuple["re.Pattern[str]", int]]] = {
    "db_connection_pool_exhaustion": [
        (_P(r"connection[- ]pool", _I), 3),
        (_P(r"pool (?:is )?(?:exhaust|full|saturat)", _I), 3),
        (_P(r"QueuePool|HikariPool|PoolTimeout", _I), 3),
        (_P(r"too many (?:connections|clients)", _I), 3),
        (_P(r"remaining connection slots", _I), 3),
        (_P(r"MongoServerSelectionError", _I), 3),
        (_P(r"max(?:imum)?[_ ]?pool[_ ]?size|pool_size|max_connections", _I), 2),
        (_P(r"connection (?:utili[sz]ation|leak|not (?:being )?released)", _I), 2),
        (_P(r"timeout waiting for (?:a )?connection", _I), 3),
        (_P(r"\b503s?\b", _I), 1),
    ],
    "auth_token_expiry": [
        (_P(r"token (?:refresh|validation|expir|signature|rejected)", _I), 3),
        (_P(r"(?:expired|invalid|malformed) (?:access |refresh |jwt )?token", _I), 3),
        (_P(r"\bjwt\b|\bjwks\b|signing key|key rotation", _I), 3),
        (_P(r"\b401s?\b|unauthori[sz]ed", _I), 2),
        (_P(r"session (?:expired|invalid|store)", _I), 2),
        (_P(r"\boauth\b|\boidc\b|\bsso\b|\bsaml\b", _I), 2),
        (_P(r"clock skew|\bnbf\b|\bexp\b claim", _I), 3),
        (_P(r"\b403\b|forbidden|login (?:fail|error)", _I), 1),
        (_P(r"auth(?:entication|orization)?", _I), 1),
    ],
    "memory_leak": [
        (_P(r"oom[- ]?kill|OOMKilled|out of memory|OutOfMemory", _I), 3),
        (_P(r"memory (?:leak|climb|grow|keeps growing|creep)", _I), 3),
        (_P(r"heap (?:usage|size|dump|exhaust|space)", _I), 3),
        (_P(r"memory (?:usage|pressure|limit|utili[sz]ation)", _I), 2),
        (_P(r"\brss\b|gc overhead|garbage collect|full gc", _I), 2),
        (_P(r"restarts? (?:temporarily )?(?:fix|resolve)", _I), 2),
        (_P(r"exit code 137|\b137\b", _I), 2),
    ],
    "rate_limit_breach": [
        (_P(r"\b429s?\b|too many requests", _I), 3),
        (_P(r"rate[- ]?limit", _I), 3),
        (_P(r"throttl", _I), 2),
        (_P(r"quota (?:exceeded|exhausted|breach)", _I), 3),
        (_P(r"retry-after", _I), 3),
        (_P(r"request queue (?:is )?backing up|retry storm", _I), 2),
    ],
    "bad_deploy_config": [
        (_P(r"(?:after|since|following|right after) (?:a |the |our |latest |last )?(?:\w+ )?(?:deploy|release|rollout|migration|upgrade)", _I), 3),
        (_P(r"rollback|rolled back|roll back", _I), 3),
        (_P(r"env(?:ironment)? var(?:iable)?s?|configmap|helm (?:chart|values)|feature flag", _I), 3),
        (_P(r"config(?:uration)? (?:change|diff|drift|error|mismatch|missing|typo)", _I), 3),
        (_P(r"missing (?:env|config|secret|setting)", _I), 3),
        (_P(r"regression|bad (?:release|build|config)|canary", _I), 2),
        (_P(r"deploy(?:ment|ed)?", _I), 1),
    ],
    "disk_full": [
        (_P(r"no space left|ENOSPC", _I), 3),
        (_P(r"disk (?:is )?(?:full|usage|space|pressure|quota)", _I), 3),
        (_P(r"inodes?\b|volume (?:is )?full|filesystem (?:is )?full", _I), 3),
        (_P(r"\bdf -h\b|write failed|cannot (?:write|create file)", _I), 2),
        (_P(r"log(?:s)? (?:filled|fill|rotation)|/var/log|/tmp", _I), 1),
    ],
    "cpu_saturation": [
        (_P(r"cpu (?:usage|utili[sz]ation|saturat|spike|pegged|throttl|at \d+)", _I), 3),
        (_P(r"high cpu|100% cpu|cpu[- ]bound", _I), 3),
        (_P(r"load average|run queue", _I), 3),
        (_P(r"busy[- ]loop|hot loop|runaway (?:process|thread|loop)|infinite loop", _I), 3),
        (_P(r"thread(?:s| pool)? (?:starv|exhaust|saturat)", _I), 2),
    ],
    "slow_query_db_lock": [
        (_P(r"slow quer", _I), 3),
        (_P(r"deadlock|lock wait|lock timeout|row lock|table lock|blocked quer", _I), 3),
        (_P(r"missing index|full table scan|seq(?:uential)? scan|explain analyze", _I), 3),
        (_P(r"long[- ]running (?:query|transaction)|idle in transaction", _I), 3),
        (_P(r"replication lag|replica lag", _I), 3),
        (_P(r"statement timeout|query (?:time|latency|duration)", _I), 2),
    ],
    "cache_failure": [
        (_P(r"\bredis\b|memcache", _I), 3),
        (_P(r"cache (?:miss|stampede|eviction|hit[- ]rate|cluster|node|is down|outage)", _I), 3),
        (_P(r"thundering herd|dogpile|evicted_keys|maxmemory", _I), 3),
        (_P(r"\bREADONLY\b|cache-aside|cold cache", _I), 2),
    ],
    "queue_backlog": [
        (_P(r"consumer lag|lag (?:is )?(?:growing|increasing|high)|offset lag", _I), 3),
        (_P(r"kafka|rabbitmq|\bsqs\b|pub/?sub|celery|sidekiq|kinesis", _I), 3),
        (_P(r"backlog|queue (?:depth|length|size|is growing|growing)", _I), 3),
        (_P(r"dead[- ]letter|\bdlq\b|rebalanc|message age", _I), 3),
        (_P(r"poison (?:pill|message)", _I), 3),
    ],
    "tls_certificate_expiry": [
        (_P(r"certificate (?:has )?(?:expired|expiry|expiration|invalid|verify)|cert(?:s)? expir", _I), 3),
        (_P(r"x509|CERTIFICATE_VERIFY_FAILED|ERR_CERT", _I), 3),
        (_P(r"(?:tls|ssl) (?:handshake|error|failure|alert)|handshake (?:fail|error)", _I), 3),
        (_P(r"\btls\b|\bssl\b|\bcertificate\b", _I), 1),
        (_P(r"let'?s ?encrypt|cert-manager|acme", _I), 2),
    ],
    "dns_network_failure": [
        (_P(r"\bdns\b|nxdomain|getaddrinfo|ENOTFOUND|name resolution", _I), 3),
        (_P(r"ECONNREFUSED|ECONNRESET|ETIMEDOUT|connection (?:refused|reset by peer)", _I), 3),
        (_P(r"no route to host|packet loss|network partition|unreachable", _I), 3),
        (_P(r"connection timed out|i/o timeout", _I), 2),
        (_P(r"security group|firewall|network polic|\bvpc\b|load balancer", _I), 1),
    ],
    "dependency_outage": [
        (_P(r"third[- ]party|upstream (?:provider|service|api|outage|timeout|error)|vendor", _I), 3),
        (_P(r"bad gateway|gateway timeout|\b502\b|\b504\b", _I), 2),
        (_P(r"circuit[- ]breaker|(?:downstream|upstream) dependency", _I), 3),
        (_P(r"provider (?:outage|incident|degrad)|status ?page", _I), 3),
        (_P(r"stripe|twilio|sendgrid|paypal|aws (?:outage|region)|cloud provider", _I), 2),
    ],
    "k8s_pod_crashloop": [
        (_P(r"crash ?loop ?back ?off", _I), 3),
        (_P(r"image ?pull ?back ?off|ErrImagePull", _I), 3),
        (_P(r"pods? (?:is |are )?(?:restarting|evicted|pending|crash|not ready)", _I), 3),
        (_P(r"liveness|readiness probe|probe failed", _I), 3),
        (_P(r"failed ?scheduling|insufficient (?:cpu|memory)|node not ?ready", _I), 3),
        (_P(r"kubelet|kubectl|exit code \d+", _I), 1),
    ],
    "capacity_saturation": [
        (_P(r"traffic (?:spike|surge|burst|increase)|sudden (?:spike|surge) in (?:traffic|requests)", _I), 3),
        (_P(r"auto[- ]?scal|\bhpa\b|scale[- ]out|scaling (?:limit|lag)", _I), 3),
        (_P(r"load test|flash sale|marketing campaign|black friday|peak (?:load|traffic)", _I), 3),
        (_P(r"requests per second|\brps\b|\bqps\b", _I), 2),
        (_P(r"saturat|capacity", _I), 1),
    ],
}

# Loose aliases so an engineer-typed slug ("db pool exhaustion", "oom") still
# maps onto a real category.
_SLUG_ALIASES: dict[str, str] = {
    "db_pool": "db_connection_pool_exhaustion",
    "connection_pool": "db_connection_pool_exhaustion",
    "pool_exhaustion": "db_connection_pool_exhaustion",
    "auth": "auth_token_expiry",
    "token_expiry": "auth_token_expiry",
    "oom": "memory_leak",
    "memory": "memory_leak",
    "rate_limit": "rate_limit_breach",
    "throttling": "rate_limit_breach",
    "bad_deploy": "bad_deploy_config",
    "config_error": "bad_deploy_config",
    "disk": "disk_full",
    "cpu": "cpu_saturation",
    "slow_query": "slow_query_db_lock",
    "deadlock": "slow_query_db_lock",
    "redis": "cache_failure",
    "cache": "cache_failure",
    "kafka_lag": "queue_backlog",
    "queue": "queue_backlog",
    "cert_expiry": "tls_certificate_expiry",
    "tls": "tls_certificate_expiry",
    "dns": "dns_network_failure",
    "network": "dns_network_failure",
    "upstream": "dependency_outage",
    "third_party": "dependency_outage",
    "crashloop": "k8s_pod_crashloop",
    "traffic_spike": "capacity_saturation",
}

MIN_CONFIDENT_SCORE = 3  # below this a category is "weak" and never trusted alone


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(text: str, service: str = "") -> list[tuple[str, float]]:
    """Rank categories for an incident. Returns [(slug, score)] best first,
    only categories with score > 0. Symptom text dominates; the service name
    contributes a little (e.g. ``redis-proxy``)."""
    blob = f"{text or ''}\n{service or ''}"
    scores: dict[str, float] = {}
    for slug, patterns in _PATTERNS.items():
        total = 0.0
        for rx, weight in patterns:
            found = rx.findall(blob)
            if found:
                # diminishing returns for repeated hits of the same pattern
                total += weight * (1 + 0.25 * min(len(found) - 1, 3))
        if total > 0:
            scores[slug] = total
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


def normalize_family(slug: Optional[str]) -> Optional[str]:
    """Map a free-form root-cause slug onto a taxonomy category when possible.
    Returns the original (cleaned) slug for custom families, None when empty."""
    if not slug:
        return None
    clean = re.sub(r"[^a-z0-9]+", "_", str(slug).lower()).strip("_")
    if not clean:
        return None
    if clean in CATEGORY_LABELS:
        return clean
    if clean in _SLUG_ALIASES:
        return _SLUG_ALIASES[clean]
    ranked = classify(clean.replace("_", " "))
    if ranked and ranked[0][1] >= MIN_CONFIDENT_SCORE:
        return ranked[0][0]
    return clean


def label_for(slug: Optional[str]) -> str:
    if not slug:
        return "Unclassified incident"
    return CATEGORY_LABELS.get(slug) or slug.replace("_", " ").capitalize()


# ---------------------------------------------------------------------------
# Signal extraction (concrete evidence from the incident text)
# ---------------------------------------------------------------------------

_SIGNAL_RES: list["re.Pattern[str]"] = [
    _P(r"\bHTTP\s*[45]\d\d\b", _I),
    _P(r"\b(?:401|403|429|500|502|503|504)s?\b", _I),
    _P(r"\b[45]\d\d\b(?=\s+(?:responses?|errors?|status|from|after|on))", _I),
    _P(r"\b\d{1,3}(?:\.\d+)?\s?%(?:\s+(?:of\s+)?[a-z]+){0,2}", _I),
    _P(r"\b[A-Z][A-Za-z]+(?:Error|Exception|Timeout|Exhausted)\b"),
    _P(r"\b(?:OOMKilled|CrashLoopBackOff|ImagePullBackOff|ENOSPC|ECONNREFUSED|ECONNRESET|ETIMEDOUT|NXDOMAIN)\b"),
    _P(r"no space left on device", _I),
    _P(r"connection pool (?:is )?exhausted|QueuePool limit[^.\n]{0,30}", _I),
    _P(r"(?:token|jwt|certificate|session)[a-z ]{0,20}(?:expired|invalid|failed|failure|rejected)", _I),
    _P(r"\b\d+(?:\.\d+)?\s?(?:ms|s|sec|seconds|min|minutes)\b(?=\s+(?:latency|timeout|delay|p9\d))", _I),
    _P(r"p9\d\s+latency[^.\n]{0,25}", _I),
    _P(r"consumer lag[^.\n]{0,25}", _I),
    _P(r"after (?:the |a |our )?(?:latest |last )?[a-z ]{0,15}(?:deploy(?:ment)?|release|rollout)", _I),
]


def extract_signals(text: str, limit: int = 8) -> list[str]:
    """Short, human-readable evidence fragments found in the incident text."""
    seen: set[str] = set()
    out: list[str] = []
    for rx in _SIGNAL_RES:
        for m in rx.finditer(text or ""):
            frag = re.sub(r"\s+", " ", m.group(0)).strip(" .,:;\"'")
            key = frag.lower()
            if len(frag) < 3 or key in seen or any(key in k for k in seen):
                continue
            seen.add(key)
            out.append(frag[:60])
            if len(out) >= limit:
                return out
    return out


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

_STOP = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it its
may might not of on or our out over should so some such than that the their then there these they this those
to too up us was we were what when where which while who will with would you your
service services api apis app application server servers system systems request requests response responses
error errors issue issues incident incidents problem users user seeing seen see showing shows show logs log
intermittent intermittently started shortly latest recent recently currently again also after before during
returning returned return received receiving reported report happening happen occur occurs occurred
production prod http https com org new old one two three high low
""".split())

_TOK_RE = re.compile(r"[a-z][a-z0-9]{2,}")


def _stem(tok: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if tok.endswith(suf) and len(tok) - len(suf) >= 4:
            return tok[: -len(suf)]
    return tok


def tokens(text: str) -> set[str]:
    return {
        _stem(t) for t in _TOK_RE.findall((text or "").lower().replace("_", " "))
        if t not in _STOP
    }


def similarity(a: str, b: str) -> float:
    """Overlap coefficient on content tokens, damped for tiny overlaps.
    0.0 (nothing shared) .. 1.0 (one text contained in the other)."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    if inter < 2:
        return 0.0
    overlap = inter / min(len(ta), len(tb))
    cosine = inter / math.sqrt(len(ta) * len(tb))
    return round(min(1.0, (0.6 * overlap + 0.4 * cosine)) * min(1.0, inter / 4), 4)


# ---------------------------------------------------------------------------
# Incident profile + topical gate
# ---------------------------------------------------------------------------

# A memory in a DIFFERENT category needs at least this much textual overlap to
# still count (e.g. an auth incident that genuinely also mentions pool errors).
CROSS_CATEGORY_MIN_SIM = 0.45
# When either side has no trustworthy category, fall back to text similarity.
UNCATEGORIZED_MIN_SIM = 0.14


@dataclass
class Profile:
    """Deterministic read of the CURRENT incident."""
    service: str
    text: str
    ranked: list[tuple[str, float]] = field(default_factory=list)
    signals: list[str] = field(default_factory=list)

    @property
    def primary(self) -> Optional[str]:
        if self.ranked and self.ranked[0][1] >= MIN_CONFIDENT_SCORE:
            return self.ranked[0][0]
        return None

    @property
    def secondary(self) -> Optional[str]:
        if self.primary and len(self.ranked) > 1:
            slug, score = self.ranked[1]
            if score >= MIN_CONFIDENT_SCORE and score >= 0.6 * self.ranked[0][1]:
                return slug
        return None

    @property
    def related(self) -> set[str]:
        return {c for c in (self.primary, self.secondary) if c}

    @property
    def confidence_pct(self) -> float:
        """How sure the classifier is (share of the top score, capped)."""
        if not self.ranked:
            return 0.0
        top = self.ranked[0][1]
        total = sum(s for _, s in self.ranked[:3]) or 1.0
        strength = min(1.0, top / 9.0)
        dominance = top / total
        return round(100 * (0.6 * strength + 0.4 * dominance), 1)

    def query_terms(self) -> str:
        """Extra words appended to memory searches so recall is anchored on
        THIS incident rather than on whatever the model typed."""
        parts: list[str] = []
        if self.primary:
            parts.append(label_for(self.primary))
        parts.extend(self.signals[:3])
        return " ".join(parts)


def build_profile(service: str, symptom: str) -> Profile:
    text = f"{service}\n{symptom}"
    return Profile(
        service=service or "",
        text=text,
        ranked=classify(symptom, service),
        signals=extract_signals(symptom),
    )


def category_of_text(text: str, declared_family: Optional[str] = None) -> Optional[str]:
    """Category of a memory: its declared root-cause family when that maps onto
    the taxonomy, otherwise whatever its content classifies as (if confident)."""
    fam = normalize_family(declared_family)
    if fam and fam in CATEGORY_LABELS:
        return fam
    ranked = classify(text)
    if ranked and ranked[0][1] >= MIN_CONFIDENT_SCORE:
        return ranked[0][0]
    return None


def topical_verdict(profile: Profile, memory_text: str, declared_family: Optional[str] = None) -> tuple[bool, str, float]:
    """May this memory count as evidence for the current incident?

    Returns (ok, reason, similarity). Rules:
      * memory category is the incident's primary/secondary category → yes
      * categories known on both sides but different → only with strong text overlap
      * a category unknown on either side → text similarity must clear a floor
    """
    sim = similarity(profile.text, memory_text)
    mem_cat = category_of_text(memory_text, declared_family)
    cur = profile.related
    if cur and mem_cat:
        if mem_cat in cur:
            return True, f"same category ({label_for(mem_cat)})", max(sim, 0.5)
        if sim >= CROSS_CATEGORY_MIN_SIM:
            return True, f"different category but strong text overlap ({sim:.2f})", sim
        return False, (
            f"different problem: memory is {label_for(mem_cat)}, "
            f"incident looks like {label_for(profile.primary)}"
        ), sim
    if sim >= UNCATEGORIZED_MIN_SIM:
        return True, f"textual overlap ({sim:.2f})", sim
    return False, f"no topical overlap (similarity {sim:.2f})", sim
