"""
Developer playbooks — problem -> solution recommendations per incident category.

The LLM writes the free-text diagnosis; this module guarantees that every
incident ALSO gets concrete, category-specific engineering guidance:

    problem  ->  likely causes  ->  do now  ->  investigate (commands)
             ->  permanent fix  ->  verify  ->  prevent

Recommendations are built from the deterministic classification of the current
incident (``taxonomy.Profile``), personalised with the incident's own service
name and extracted signals, and enriched with fixes that engineers actually
confirmed in earlier, topically-matching resolved incidents.

Nothing here executes anything. Commands are read-only inspection unless the
step text says otherwise; mutating steps carry an explicit approval caution.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from . import store, taxonomy
from .schemas import CommandHint, MemoryHit, ProvenFix, Recommendation


@dataclass(frozen=True)
class Playbook:
    severity: str                      # critical | high | medium
    title: str
    problem: str                       # {service} is substituted
    mechanism: str                     # WHY it happens (semantic-layer content)
    applies_when: str                  # when this playbook is the right one
    confused_with: tuple[str, ...]
    likely_causes: tuple[str, ...]
    immediate: tuple[str, ...]
    investigate: tuple[tuple[str, str], ...]   # (what it tells you, command)
    fix: tuple[str, ...]
    verify: tuple[str, ...]
    prevent: tuple[str, ...]
    rollback: str = ""
    caution: str = ""


PLAYBOOKS: dict[str, Playbook] = {
    "db_connection_pool_exhaustion": Playbook(
        severity="critical",
        title="Database connection pool exhausted — requests queue, then fail",
        problem=(
            "{service} has run out of database connections. Requests wait for a free "
            "connection, hit the acquire timeout and surface as 5xx/timeouts while the "
            "database itself may look healthy."
        ),
        mechanism=(
            "Every request borrows a connection from a fixed-size pool. When connections are "
            "held longer than they are released — leaked handles, slow queries, transactions "
            "left open, or traffic above pool capacity — the pool empties, new requests block "
            "on acquire and time out, and retries amplify the load."
        ),
        applies_when="5xx/timeouts under load, 'pool exhausted' or 'too many connections' errors, DB CPU low but app latency high.",
        confused_with=("slow_query_db_lock", "capacity_saturation"),
        likely_causes=(
            "Connections not released (missing close/finally, unhandled exception paths, leaked cursors).",
            "Slow queries or long transactions holding connections open.",
            "Pool size too small for current concurrency, or too many app replicas x pool size versus DB max_connections.",
            "Retry storm after a brief DB blip multiplying demand for connections.",
        ),
        immediate=(
            "Check pool utilisation vs. limit right now (app metrics or DB connection count) to confirm exhaustion.",
            "Shed load: lower concurrency / enable rate limiting on the heaviest endpoints while you investigate.",
            "If a specific replica is stuck at 100% pool use, restart that pod only (with on-call approval) to release leaked connections.",
        ),
        investigate=(
            ("Connections by state/client on the DB", "SELECT state, count(*), max(now()-state_change) FROM pg_stat_activity GROUP BY state;  -- Postgres"),
            ("Longest-held connections / open transactions", "SELECT pid, now()-xact_start AS age, query FROM pg_stat_activity WHERE xact_start IS NOT NULL ORDER BY age DESC LIMIT 10;  -- Postgres"),
            ("Mongo current connections vs available", "db.serverStatus().connections  // mongosh"),
            ("Recent app errors mentioning the pool", "kubectl logs deploy/{service} --since=30m | grep -i -E 'pool|too many connections|timeout'"),
        ),
        fix=(
            "Fix leaks: ensure every acquire is released in a finally/context-manager path; add a pool 'checked-out' gauge and leak-detection timeout.",
            "Right-size the pool: replicas x maxPoolSize must stay below the DB's max connections with headroom; raise maxPoolSize only after confirming DB capacity.",
            "Set sane acquire/idle/max-lifetime timeouts and cap statement/transaction time so slow work cannot pin connections.",
            "Put a pooler (PgBouncer/ProxySQL) in front of the DB if many short-lived services connect.",
        ),
        verify=(
            "Pool in-use stays well below max (target < 70%) under normal traffic.",
            "5xx rate and p95 latency return to baseline and stay there for 30+ minutes.",
            "DB connection count is flat rather than climbing after deploys/restarts.",
        ),
        prevent=(
            "Alert on pool utilisation > 80% and on acquire-wait time, not just on 5xx.",
            "Load-test after any change to pool config, replica count or query patterns.",
            "Add retry with jittered backoff and a circuit breaker so blips don't become retry storms.",
        ),
        rollback="If the incident began right after a deploy touching data access, roll that deploy back first.",
        caution="Do not blindly raise pool size — that can push the database itself over its connection limit.",
    ),
    "auth_token_expiry": Playbook(
        severity="high",
        title="Authentication failures — tokens rejected or refresh failing",
        problem=(
            "Clients of {service} are getting 401/unauthorized responses: access tokens are "
            "being rejected or the token-refresh flow is failing or timing out, forcing users "
            "to re-authenticate."
        ),
        mechanism=(
            "Token validation depends on three things staying consistent across services: the "
            "signing keys, the clock, and the expiry/refresh contract. A deploy that changes "
            "refresh logic, key rotation that isn't propagated, clock skew, or a slow/unavailable "
            "identity provider makes valid sessions look invalid and produces intermittent 401s."
        ),
        applies_when="Spurious 401s, 'invalid/expired token', refresh calls failing or timing out, often right after an auth-related deploy.",
        confused_with=("bad_deploy_config", "dependency_outage"),
        likely_causes=(
            "Recent deploy changed token-refresh, expiry (TTL) or validation logic.",
            "Signing key / JWKS rotated but some validators still cache the old key.",
            "Clock skew between issuer and validators (nbf/exp claims fail).",
            "Identity provider or session store slow/unavailable, so refresh requests time out.",
        ),
        immediate=(
            "Confirm scope: are 401s tied to one client/version/region or global? Compare failing vs. passing requests.",
            "If it began right after an auth deploy, prepare (and get approval for) a rollback of that release.",
            "Temporarily raise validator JWKS cache refresh / extend token grace period only if security policy allows.",
        ),
        investigate=(
            ("401 rate and timing vs. deploys", "kubectl rollout history deploy/{service}  # compare revision times with first 401"),
            ("Token validation / refresh errors", "kubectl logs deploy/{service} --since=30m | grep -i -E 'token|jwt|401|refresh|signature|expired'"),
            ("Clock drift on nodes", "kubectl get nodes -o name | xargs -I{} kubectl debug {} -it --image=busybox -- date  # compare with a trusted source"),
            ("Inspect a failing token's claims (never share it)", "echo $TOKEN | cut -d. -f2 | base64 -d 2>/dev/null | jq '{iss,aud,exp,nbf,iat}'"),
        ),
        fix=(
            "Fix the refresh/validation regression and add a contract test covering expiry, refresh and clock-skew tolerance (allow 30-60s leeway).",
            "Make key rotation safe: publish new keys before signing with them, keep old keys until all tokens signed by them expire.",
            "Add timeouts, retries with backoff and a fallback for the identity-provider/session-store call in the refresh path.",
        ),
        verify=(
            "401 rate returns to baseline across all clients and regions.",
            "Successful token refresh rate and refresh latency are back to normal.",
            "A test login → use → refresh → use cycle passes end to end.",
        ),
        prevent=(
            "Canary auth deploys and gate on 401/refresh-failure rate.",
            "Alert on sudden 401 rate change and on JWKS fetch failures.",
            "Run NTP/chrony health checks on all nodes.",
        ),
        rollback="Roll back the auth deployment to the previous known-good revision if 401s started with it.",
        caution="Do not disable token validation to 'get traffic flowing' — that is a security incident of its own.",
    ),
    "memory_leak": Playbook(
        severity="high",
        title="Memory growth until OOM-kill — recurring restarts",
        problem=(
            "{service} memory climbs steadily until the container is OOM-killed, restarts, and "
            "the cycle repeats. Restarts hide the problem temporarily but cause dropped requests."
        ),
        mechanism=(
            "Objects stay reachable after they are no longer needed — unbounded caches, "
            "listeners/timers never removed, growing in-memory queues, unclosed streams — so heap "
            "grows monotonically. Once RSS crosses the container memory limit the kernel OOM-kills "
            "the process; the restart resets memory and the leak starts over."
        ),
        applies_when="Sawtooth memory graph, OOMKilled / exit code 137, restarts that temporarily fix things, recurrence every few hours.",
        confused_with=("cpu_saturation", "k8s_pod_crashloop"),
        likely_causes=(
            "Unbounded in-process cache or map keyed by request/user data.",
            "Event listeners, timers, or goroutines/threads created per request and never released.",
            "Large payloads buffered fully in memory (uploads, batch jobs, queue messages).",
            "Memory limit set too low for the workload, or a recent change increased footprint.",
        ),
        immediate=(
            "Confirm it is OOMKilled (not a crash) and note the restart interval to estimate leak rate.",
            "Stagger/rolling-restart pods before they hit the limit to stop simultaneous failures (with approval).",
            "Temporarily raise the memory limit only as a stop-gap; note it as debt, not a fix.",
        ),
        investigate=(
            ("Was the last termination an OOM kill?", "kubectl get pod -l app={service} -o jsonpath='{range .items[*]}{.metadata.name}{\" \"}{.status.containerStatuses[0].lastState.terminated.reason}{\"\\n\"}{end}'"),
            ("Memory usage per pod over time", "kubectl top pod -l app={service} --containers"),
            ("Capture a heap profile/dump before the next OOM", "# JVM: jcmd <pid> GC.heap_dump /tmp/heap.hprof   |  Node: node --heapsnapshot-signal=SIGUSR2   |  Go: curl :6060/debug/pprof/heap"),
            ("Correlate growth with a release", "kubectl rollout history deploy/{service}"),
        ),
        fix=(
            "Analyse the heap dump/profile, find the dominant retained object type and the code path that allocates it; fix the retention.",
            "Bound every cache/queue (max size + TTL/eviction) and stream large payloads instead of buffering.",
            "Set requests/limits from measured usage with headroom, and tune GC/heap flags to the container limit.",
        ),
        verify=(
            "Memory graph is flat (or bounded sawtooth from GC) for 24h+ with no OOMKilled restarts.",
            "Restart count for the deployment stops increasing.",
        ),
        prevent=(
            "Alert on memory slope / >80% of limit, not only on OOM events.",
            "Add a soak test to CI/staging that fails on monotonic memory growth.",
        ),
        rollback="If growth started with a specific release, roll back while the leak is analysed offline.",
    ),
    "rate_limit_breach": Playbook(
        severity="high",
        title="Rate limit / throttling (HTTP 429) cascading into request backlog",
        problem=(
            "{service} is receiving 429 / throttling responses from a dependency (or from its own "
            "limiter), and retries are piling up into a growing request queue."
        ),
        mechanism=(
            "When request volume exceeds a quota, the provider answers 429. Clients that retry "
            "immediately and in lockstep add load exactly when capacity is scarce (retry storm), "
            "so queues grow, latency rises and the failure spreads upstream."
        ),
        applies_when="Spikes of 429 / Retry-After, quota errors, queues backing up behind a throttled dependency.",
        confused_with=("dependency_outage", "capacity_saturation"),
        likely_causes=(
            "Traffic or batch job above the provider's/own quota.",
            "Retries without backoff/jitter (retry storm) or missing Retry-After handling.",
            "Many instances sharing one API key/quota with no coordination.",
            "Quota reduced or plan changed on the provider side.",
        ),
        immediate=(
            "Respect Retry-After: switch retries to exponential backoff with jitter and cap concurrency to the dependency.",
            "Pause or slow non-critical batch/backfill traffic that shares the quota.",
            "Prioritise critical traffic (e.g. payments) over bulk work.",
        ),
        investigate=(
            ("429 rate over time and by caller", "kubectl logs deploy/{service} --since=30m | grep -c ' 429 '"),
            ("Retry-After values being returned", "kubectl logs deploy/{service} --since=30m | grep -i 'retry-after' | tail -20"),
            ("Request queue depth trend", "# check your queue/worker dashboard for depth and age of oldest job"),
        ),
        fix=(
            "Implement client-side rate limiting (token bucket) sized below the quota, plus backoff with jitter and a retry budget.",
            "Add a circuit breaker and a bounded queue with load-shedding so throttling can't cause unbounded backlog.",
            "Request a higher quota or split traffic across keys/accounts where the provider's terms allow.",
        ),
        verify=(
            "429 rate falls near zero and queue depth/age drain back to baseline.",
            "No retry storm pattern (retries/second) after the fix under a replayed spike.",
        ),
        prevent=(
            "Alert at 70% of quota consumption, not after 429s begin.",
            "Cache or batch calls to the throttled API where possible.",
        ),
    ),
    "bad_deploy_config": Playbook(
        severity="high",
        title="Regression after deploy / configuration change",
        problem=(
            "Errors on {service} started immediately after a deploy or configuration change; "
            "the timing points at the release rather than at load or infrastructure."
        ),
        mechanism=(
            "A release changed code, an environment variable, a feature flag or a config value "
            "that the running system depends on. Because the change is applied everywhere at once, "
            "the error rate steps up right at the rollout time, and reverting the change removes it."
        ),
        applies_when="Error-rate step change aligned with a deploy/config change; rollback fixes it; config diff shows a changed variable.",
        confused_with=("auth_token_expiry", "dependency_outage"),
        likely_causes=(
            "Changed/missing environment variable, secret or ConfigMap value.",
            "Feature flag flipped or defaulted incorrectly.",
            "Incompatible code/schema/dependency version shipped in the release.",
            "Partial rollout leaving mixed versions that disagree on a contract.",
        ),
        immediate=(
            "Line up first-error timestamp with the deploy/config change time to confirm correlation.",
            "Roll back to the last known-good revision (with approval) — fastest way to stop the impact.",
            "Freeze further deploys until the cause is identified.",
        ),
        investigate=(
            ("Deployment history", "kubectl rollout history deploy/{service}"),
            ("What changed between revisions", "kubectl rollout history deploy/{service} --revision=<N> ; diff against <N-1>"),
            ("Effective env vars in the running pod", "kubectl exec deploy/{service} -- env | sort"),
            ("Startup/runtime errors after rollout", "kubectl logs deploy/{service} --since=30m | grep -i -E 'error|missing|invalid|config'"),
        ),
        fix=(
            "Correct the offending config/code, add validation so a bad or missing value fails fast at startup (readiness) instead of at runtime.",
            "Add the failing scenario as a regression/integration test and a config-schema check in CI.",
            "Ship the fix through canary → gradual rollout with automatic rollback on error-rate increase.",
        ),
        verify=(
            "Error rate returns to pre-deploy baseline on the fixed/rolled-back revision.",
            "All replicas run the same revision and pass readiness.",
        ),
        prevent=(
            "Progressive delivery (canary) with SLO-based auto-rollback.",
            "Config diff review + schema validation in the deploy pipeline.",
        ),
        rollback="kubectl rollout undo deploy/{service}  (requires on-call approval)",
    ),
    "disk_full": Playbook(
        severity="critical",
        title="Disk / storage exhausted — writes failing",
        problem=(
            "{service} cannot write because the filesystem (or inode table) is full: uploads, "
            "logs, temp files or database writes fail with 'No space left on device'."
        ),
        mechanism=(
            "Something writes faster than it is cleaned up — unrotated logs, temp files from failed "
            "jobs, uploads never garbage-collected, WAL/journal growth. Once free blocks or inodes "
            "reach zero every write fails, and processes that depend on writing (DBs, loggers) "
            "start failing too."
        ),
        applies_when="'No space left on device' / ENOSPC, df at 100%, failures on write paths, often logs or /tmp.",
        confused_with=("memory_leak",),
        likely_causes=(
            "Log files not rotated or debug logging left on.",
            "Temp/upload files not cleaned after failed or abandoned operations.",
            "Volume too small for data growth, or inodes exhausted by many tiny files.",
            "Database WAL/binlog or core dumps accumulating.",
        ),
        immediate=(
            "Identify the full mount and free space safely: delete/rotate old logs and known-safe temp files first.",
            "Do not delete database files or unknown data; if the DB volume is full, expand the volume instead.",
            "Expand the PVC/volume if cleanup is not enough (with approval).",
        ),
        investigate=(
            ("Which filesystem is full (blocks and inodes)", "kubectl exec deploy/{service} -- sh -c 'df -h; df -i'"),
            ("Largest directories", "kubectl exec deploy/{service} -- sh -c 'du -xh / 2>/dev/null | sort -rh | head -20'"),
            ("Large or growing logs", "kubectl exec deploy/{service} -- sh -c 'ls -lhS /var/log | head'"),
        ),
        fix=(
            "Configure log rotation/retention and lower log verbosity; ship logs off-node.",
            "Add lifecycle cleanup for temp/upload data (TTL job, cleanup on failure paths).",
            "Right-size the volume with headroom and enable auto-expansion where supported.",
        ),
        verify=(
            "Free space > 30% and stable; inode usage healthy.",
            "Write-path errors (uploads, logging, DB writes) stop.",
        ),
        prevent=(
            "Alert on disk > 75% and on growth rate (predict time-to-full).",
            "Set ephemeral-storage requests/limits so one pod can't fill a node.",
        ),
        caution="Never delete files you cannot identify; wrong deletions on a data volume can cause permanent loss.",
    ),
    "cpu_saturation": Playbook(
        severity="high",
        title="CPU saturation — latency climbs, requests time out",
        problem=(
            "{service} is CPU-bound: cores are pegged or throttled, so request handling slows, "
            "queues form and timeouts follow."
        ),
        mechanism=(
            "When runnable work exceeds available CPU (hot loop, expensive endpoint, regex/JSON "
            "blow-up, GC thrash, or CPU limits throttling the container), the scheduler time-slices "
            "everything and latency grows non-linearly as utilisation approaches 100%."
        ),
        applies_when="CPU near 100% or throttled, rising latency, load average high, often after a code/traffic change.",
        confused_with=("capacity_saturation", "memory_leak"),
        likely_causes=(
            "Hot loop or expensive code path introduced by a release.",
            "Traffic increase without matching scale-out.",
            "GC thrashing (high memory pressure) burning CPU.",
            "CPU limit too low causing throttling.",
        ),
        immediate=(
            "Scale out replicas (or raise HPA max) to buy headroom (with approval).",
            "Identify and shed/disable the most expensive endpoint or job.",
        ),
        investigate=(
            ("Per-pod CPU and throttling", "kubectl top pod -l app={service} --containers"),
            ("CPU throttling ratio", "# PromQL: rate(container_cpu_cfs_throttled_periods_total{pod=~\"{service}.*\"}[5m])"),
            ("Where CPU is spent", "# profile: py-spy top --pid <pid>  |  perf top  |  Go: pprof cpu  |  JVM: async-profiler"),
        ),
        fix=(
            "Fix the hot path found by profiling (algorithm, N+1 work, regex, serialization).",
            "Tune requests/limits and HPA (target ~60% CPU) so scale-out precedes saturation.",
        ),
        verify=(
            "CPU utilisation and throttling return to normal; p95 latency back to baseline.",
        ),
        prevent=(
            "Alert on CPU > 80% sustained and on throttling.",
            "Add performance regression checks for hot endpoints.",
        ),
    ),
    "slow_query_db_lock": Playbook(
        severity="high",
        title="Slow queries / lock contention in the database",
        problem=(
            "{service} is slow or timing out because database queries are slow, blocked on "
            "locks, or waiting behind long-running transactions."
        ),
        mechanism=(
            "A missing index, a plan regression, or a long transaction makes queries hold locks or "
            "resources longer. Other queries queue behind them, connections stay busy, and latency "
            "compounds until timeouts or deadlocks appear."
        ),
        applies_when="Slow query logs, lock waits/deadlocks, replication lag, latency rising with DB time dominating.",
        confused_with=("db_connection_pool_exhaustion",),
        likely_causes=(
            "Missing/unused index after a schema or query change.",
            "Long-running or idle-in-transaction sessions holding locks.",
            "Large batch update/migration locking hot tables.",
            "Replica lag causing stale reads/timeouts.",
        ),
        immediate=(
            "Find blocking sessions and the offending query; terminate a clearly runaway session only with DBA/on-call approval.",
            "Pause the batch job/migration if it started the contention.",
        ),
        investigate=(
            ("Blocked and blocking queries", "SELECT pid, pg_blocking_pids(pid) AS blocked_by, wait_event, query FROM pg_stat_activity WHERE cardinality(pg_blocking_pids(pid))>0;  -- Postgres"),
            ("Slowest statements", "SELECT query, calls, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;  -- Postgres"),
            ("Plan of the slow query", "EXPLAIN (ANALYZE, BUFFERS) <query>;"),
        ),
        fix=(
            "Add/adjust the index or rewrite the query per the EXPLAIN plan; test on production-sized data.",
            "Break large writes into small batches and keep transactions short.",
            "Set statement_timeout / lock_timeout and idle_in_transaction_session_timeout.",
        ),
        verify=(
            "Query latency and lock-wait metrics return to baseline; no deadlock errors.",
            "Replication lag near zero.",
        ),
        prevent=(
            "Enable slow-query logging and alert on p95 query time.",
            "Review query plans for new queries in code review.",
        ),
        caution="Killing sessions and building indexes on hot tables can cause more impact — use CONCURRENTLY/online options and get approval.",
    ),
    "cache_failure": Playbook(
        severity="high",
        title="Cache failure or stampede — backend overloaded",
        problem=(
            "The cache layer used by {service} is failing, evicting too aggressively or was "
            "flushed, so requests fall through to the database/backends and overload them."
        ),
        mechanism=(
            "The cache normally absorbs most reads. When it is unavailable, cold, or many hot keys "
            "expire together (stampede/thundering herd), all requests hit the origin at once; the "
            "origin slows, timeouts rise, and recovery is slow because the cache can't refill."
        ),
        applies_when="Cache miss rate spike, Redis errors/evictions/READONLY, sudden backend load after cache restart or TTL expiry.",
        confused_with=("slow_query_db_lock", "capacity_saturation"),
        likely_causes=(
            "Cache node down/failover, network partition, or maxmemory eviction.",
            "Mass TTL expiry (same TTL for many keys) or cache flush.",
            "New code path that bypasses/invalidates the cache.",
        ),
        immediate=(
            "Confirm cache health and hit rate; if down, fail over or restart per runbook (with approval).",
            "Protect the origin: enable request coalescing/rate limiting, serve stale where safe.",
        ),
        investigate=(
            ("Redis health and evictions", "redis-cli INFO stats | grep -E 'evicted_keys|keyspace_(hits|misses)'"),
            ("Memory vs maxmemory", "redis-cli INFO memory | grep -E 'used_memory_human|maxmemory_human'"),
            ("App cache errors", "kubectl logs deploy/{service} --since=30m | grep -i -E 'redis|cache|READONLY|timeout'"),
        ),
        fix=(
            "Add TTL jitter, request coalescing (single-flight) and stale-while-revalidate to prevent stampedes.",
            "Size cache memory for the working set and choose an eviction policy suited to it.",
            "Warm critical keys after restarts/failovers.",
        ),
        verify=(
            "Hit rate returns to baseline; origin load and latency normalise.",
            "No sustained eviction or connection errors.",
        ),
        prevent=(
            "Alert on hit-rate drop and eviction rate.",
            "Load-test the cold-cache scenario.",
        ),
    ),
    "queue_backlog": Playbook(
        severity="high",
        title="Queue / consumer backlog growing",
        problem=(
            "Messages for {service} are arriving faster than they are processed, or consumers "
            "are stuck, so lag and message age keep growing."
        ),
        mechanism=(
            "Throughput of consumers falls below the arrival rate — because of slow downstream "
            "calls, a poison message being retried forever, consumer crashes/rebalances, or too few "
            "consumers — so backlog accumulates and end-to-end latency grows."
        ),
        applies_when="Consumer lag/queue depth rising, oldest message age growing, repeated rebalances, DLQ growth.",
        confused_with=("dependency_outage", "rate_limit_breach"),
        likely_causes=(
            "Poison message failing repeatedly and blocking a partition.",
            "Slow downstream dependency reducing consumer throughput.",
            "Consumers crashing or rebalancing continuously.",
            "Insufficient consumers/partitions for current volume.",
        ),
        immediate=(
            "Check for a single stuck partition/message; route it to a dead-letter queue if found.",
            "Scale consumers up to the partition count (with approval).",
        ),
        investigate=(
            ("Consumer group lag per partition", "kafka-consumer-groups.sh --bootstrap-server <broker> --describe --group <group>"),
            ("Consumer errors/rebalances", "kubectl logs deploy/{service} --since=30m | grep -i -E 'rebalanc|error|exception|retry'"),
            ("DLQ size", "# check dead-letter topic/queue depth on your broker dashboard"),
        ),
        fix=(
            "Add bounded retries with DLQ for poison messages; make handlers idempotent.",
            "Increase partitions/consumers or optimise the slow processing step.",
            "Tune poll/session timeouts to stop needless rebalances.",
        ),
        verify=(
            "Lag and oldest-message age trend to zero.",
            "DLQ not growing unexpectedly.",
        ),
        prevent=(
            "Alert on lag growth rate and message age, not only absolute depth.",
        ),
    ),
    "tls_certificate_expiry": Playbook(
        severity="critical",
        title="TLS / certificate failure — handshakes rejected",
        problem=(
            "TLS connections to or from {service} fail: a certificate expired, is not trusted, or "
            "does not match — clients see handshake/x509 errors."
        ),
        mechanism=(
            "Certificates have hard expiry dates and must chain to a trusted CA and match the "
            "hostname. When renewal automation fails or a new cert/chain is deployed incorrectly, "
            "every connection that validates the certificate is rejected at once."
        ),
        applies_when="x509 / CERTIFICATE_VERIFY_FAILED / handshake errors, sudden total failure of HTTPS calls at a specific time.",
        confused_with=("dns_network_failure", "dependency_outage"),
        likely_causes=(
            "Certificate expired because auto-renewal (cert-manager/ACME) failed.",
            "Intermediate certificate missing from the served chain.",
            "Hostname/SAN mismatch after a domain or endpoint change.",
            "Client trust store outdated.",
        ),
        immediate=(
            "Check the served certificate's expiry and chain.",
            "Renew/replace the certificate and reload the terminating proxy/ingress (with approval).",
        ),
        investigate=(
            ("Certificate dates and issuer", "echo | openssl s_client -connect <host>:443 -servername <host> 2>/dev/null | openssl x509 -noout -dates -issuer -subject"),
            ("Full chain served", "openssl s_client -connect <host>:443 -showcerts </dev/null"),
            ("cert-manager status", "kubectl get certificate,certificaterequest -A | grep -i {service}"),
        ),
        fix=(
            "Fix the renewal automation (permissions, DNS/HTTP challenge) and confirm it renews before expiry.",
            "Serve the full chain and verify SANs cover all hostnames.",
        ),
        verify=(
            "openssl shows a valid future expiry and complete chain; clients connect without errors.",
        ),
        prevent=(
            "Alert at 30/14/7 days before expiry for every certificate.",
            "Inventory certificates and owners.",
        ),
    ),
    "dns_network_failure": Playbook(
        severity="high",
        title="DNS / network connectivity failure",
        problem=(
            "{service} cannot reach a dependency: name resolution fails or connections are "
            "refused, reset or time out."
        ),
        mechanism=(
            "Requests need name resolution, a routable path and an open port. A DNS outage, changed "
            "record, network policy/security-group change or overloaded load balancer breaks one "
            "of those steps, producing refused/reset/timeout errors that look like the dependency is down."
        ),
        applies_when="ECONNREFUSED/ECONNRESET/ETIMEDOUT, NXDOMAIN/getaddrinfo errors, 'no route to host', often after infra/network change.",
        confused_with=("dependency_outage", "tls_certificate_expiry"),
        likely_causes=(
            "DNS record changed/expired or resolver (CoreDNS) overloaded.",
            "Security group / NetworkPolicy / firewall change blocking traffic.",
            "Target service down or load balancer unhealthy.",
            "Conntrack/ephemeral port exhaustion.",
        ),
        immediate=(
            "Test resolution and connectivity from inside the failing pod to separate DNS from network from the target.",
            "Revert any recent network/policy change.",
        ),
        investigate=(
            ("DNS from the pod", "kubectl exec deploy/{service} -- nslookup <dependency-host>"),
            ("TCP reachability", "kubectl exec deploy/{service} -- sh -c 'nc -vz <host> <port>'"),
            ("CoreDNS health", "kubectl -n kube-system logs -l k8s-app=kube-dns --since=15m | tail -50"),
        ),
        fix=(
            "Fix the DNS record/policy/route causing the failure and add retries with backoff plus sensible connect timeouts.",
            "Scale/tune CoreDNS and enable local DNS caching if resolver load is the cause.",
        ),
        verify=(
            "Resolution and connections succeed consistently from all pods; error rate returns to baseline.",
        ),
        prevent=(
            "Synthetic checks for DNS and key network paths; review network changes with staged rollout.",
        ),
    ),
    "dependency_outage": Playbook(
        severity="high",
        title="Upstream / third-party dependency degraded or down",
        problem=(
            "{service} is failing because an upstream dependency is returning errors (502/504, "
            "timeouts) or is unavailable."
        ),
        mechanism=(
            "When a synchronous dependency slows or fails, callers wait, exhaust threads/connections "
            "and fail themselves; without timeouts, circuit breakers and fallbacks a partial outage "
            "in one dependency spreads into a full outage of the caller."
        ),
        applies_when="502/504/timeouts to a specific upstream, provider status page shows incident, failures isolated to calls to that dependency.",
        confused_with=("dns_network_failure", "rate_limit_breach"),
        likely_causes=(
            "Provider outage or degradation.",
            "Missing/too-long timeouts letting slow calls pile up.",
            "No circuit breaker or fallback for a non-critical dependency.",
        ),
        immediate=(
            "Check the provider status page and confirm the failures are isolated to that dependency.",
            "Open the circuit / enable fallback or degraded mode for non-critical features.",
            "Tighten timeouts so slow calls fail fast and free resources.",
        ),
        investigate=(
            ("Failing upstream and latency", "kubectl logs deploy/{service} --since=30m | grep -i -E '502|504|timeout|upstream'"),
            ("Reachability of the dependency", "kubectl exec deploy/{service} -- curl -s -o /dev/null -w '%{http_code} %{time_total}s\\n' <dependency-url>"),
        ),
        fix=(
            "Add timeouts, retries with backoff, circuit breaker and graceful degradation around the dependency.",
            "Consider a secondary provider/queue-and-retry for critical flows.",
        ),
        verify=(
            "Error rate and latency recover once the dependency does; breakers close normally.",
        ),
        prevent=(
            "Dependency SLO dashboards and alerts; regular failure-injection tests.",
        ),
    ),
    "k8s_pod_crashloop": Playbook(
        severity="high",
        title="Pods crash-looping or failing to schedule/start",
        problem=(
            "Pods of {service} restart repeatedly (CrashLoopBackOff), fail probes, cannot pull "
            "their image or cannot be scheduled."
        ),
        mechanism=(
            "The container exits shortly after start (bad config, missing dependency, crash), fails "
            "its liveness/readiness probe, cannot pull the image, or the cluster lacks resources to "
            "place it. Kubernetes keeps restarting with growing back-off, so capacity shrinks."
        ),
        applies_when="CrashLoopBackOff, ImagePullBackOff, probe failures, Pending pods, node NotReady, rising restart counts.",
        confused_with=("memory_leak", "bad_deploy_config"),
        likely_causes=(
            "Application crashes at startup (config/secret/dependency missing).",
            "Liveness/readiness probe too strict or pointing at wrong path/port.",
            "Wrong image tag or registry credentials.",
            "Insufficient cluster CPU/memory or node problems.",
        ),
        immediate=(
            "Read pod events and previous-container logs to see why it exits.",
            "Roll back if it started with a new release (with approval).",
        ),
        investigate=(
            ("Why is the pod unhealthy?", "kubectl describe pod -l app={service} | sed -n '/Events:/,$p'"),
            ("Logs of the crashed container", "kubectl logs -l app={service} --previous --tail=100"),
            ("Scheduling/capacity problems", "kubectl get events --sort-by=.lastTimestamp | tail -30"),
        ),
        fix=(
            "Fix the startup error; make config validation explicit and startup fail with a clear message.",
            "Tune probes (startupProbe for slow starts) and resource requests to real usage.",
        ),
        verify=(
            "All pods Running/Ready with a stable restart count.",
        ),
        prevent=(
            "Alert on restart rate and pending pods; test images and config in staging.",
        ),
        rollback="kubectl rollout undo deploy/{service}  (requires on-call approval)",
    ),
    "capacity_saturation": Playbook(
        severity="high",
        title="Traffic surge exceeding capacity",
        problem=(
            "{service} is receiving more traffic than it can handle, so latency rises and requests "
            "fail; scaling has not kept up."
        ),
        mechanism=(
            "Capacity (replicas, threads, connections, downstream limits) is sized for normal load. "
            "A surge — campaign, retry storm, batch job — pushes utilisation toward 100% where queueing "
            "delay explodes, and autoscaling lag leaves a window of failures."
        ),
        applies_when="Requests/sec well above normal, saturation across CPU/threads/connections, correlates with an event or campaign.",
        confused_with=("cpu_saturation", "rate_limit_breach", "cache_failure"),
        likely_causes=(
            "Planned or unplanned traffic spike (campaign, viral event, bot traffic).",
            "Autoscaler max too low or scaling too slow.",
            "Retry storm multiplying real traffic.",
        ),
        immediate=(
            "Scale out now (raise replicas / HPA max) with approval.",
            "Enable load shedding/rate limits for abusive or low-priority traffic.",
        ),
        investigate=(
            ("Traffic vs. normal", "# compare current RPS with the same hour last week on your dashboard"),
            ("HPA state", "kubectl get hpa {service} -o wide"),
            ("Top callers", "kubectl logs deploy/{service} --since=15m | awk '{print $1}' | sort | uniq -c | sort -rn | head"),
        ),
        fix=(
            "Tune HPA (lower target, faster scale-up), raise limits, and pre-scale for known events.",
            "Add caching and backpressure so surges degrade gracefully.",
        ),
        verify=(
            "Latency and error rate back to baseline at current traffic; autoscaler has headroom.",
        ),
        prevent=(
            "Load-test to 2-3x expected peak; alert on saturation early.",
        ),
    ),
}

GENERIC = Playbook(
    severity="medium",
    title="Unclassified incident — structured triage",
    problem=(
        "The symptoms reported for {service} do not clearly match a known failure pattern, so "
        "start with structured triage before changing anything."
    ),
    mechanism="Root cause not yet identified; the incident text does not carry a strong signature of a known failure category.",
    applies_when="No known category matched confidently.",
    confused_with=(),
    likely_causes=(
        "A recent change (deploy, config, dependency).",
        "Resource exhaustion (CPU, memory, disk, connections).",
        "A dependency or network problem.",
    ),
    immediate=(
        "Define impact: which endpoints/users, since when, how large the error rate is.",
        "Check what changed just before the first error (deploys, config, traffic, dependencies).",
    ),
    investigate=(
        ("Recent changes", "kubectl rollout history deploy/{service}"),
        ("Recent errors", "kubectl logs deploy/{service} --since=30m | grep -i -E 'error|exception|timeout' | tail -50"),
        ("Resource usage", "kubectl top pod -l app={service}"),
    ),
    fix=(
        "Once the failing component is identified, record the root cause so future incidents are classified automatically.",
    ),
    verify=("Error rate and latency return to baseline.",),
    prevent=("Add an alert/dashboard for the signal that would have caught this earlier.",),
)


# ---------------------------------------------------------------------------
# Building recommendations
# ---------------------------------------------------------------------------

def playbook_for(category: Optional[str]) -> Playbook:
    return PLAYBOOKS.get(category or "", GENERIC)


def _svc_slug(service: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (service or "").lower()).strip("-") or "app"


def _fill(text: str, service: str) -> str:
    return text.replace("{service}", _svc_slug(service) if _looks_like_command(text) else service)


def _looks_like_command(text: str) -> bool:
    return bool(re.search(r"\b(?:kubectl|redis-cli|kafka-|openssl|echo|SELECT|db\.)", text))


def _proven_fixes(hits: list[MemoryHit], category: Optional[str]) -> list[ProvenFix]:
    """Fixes engineers CONFIRMED in earlier resolved incidents that were
    accepted as topical evidence for this incident."""
    out: list[ProvenFix] = []
    seen: set[int] = set()
    for h in hits:
        iid = h.historical_incident_id
        if iid is None or iid in seen:
            continue
        rec = store.get_incident(iid)
        if not rec or rec.get("status") != "resolved":
            continue
        fam = taxonomy.normalize_family(rec.get("root_cause_family"))
        if category and fam and fam in taxonomy.CATEGORY_LABELS and fam != category:
            continue
        try:
            import json
            steps = json.loads(rec.get("fix_steps_json") or "[]")
        except (TypeError, ValueError):
            steps = []
        if not steps:
            continue
        seen.add(iid)
        out.append(ProvenFix(
            incident_id=iid, service=rec.get("service"),
            outcome=rec.get("outcome"), steps=[str(s) for s in steps][:8],
        ))
        if len(out) >= 2:
            break
    return out


def _build_one(pb: Playbook, category: Optional[str], profile: taxonomy.Profile,
               service: str, role: str, hits: list[MemoryHit]) -> Recommendation:
    signals = profile.signals[:6]
    problem = _fill(pb.problem, service)
    if signals:
        problem += " Evidence in this incident: " + "; ".join(signals) + "."
    conf = profile.confidence_pct if role == "primary" else max(20.0, profile.confidence_pct * 0.6)
    return Recommendation(
        category=category or "unclassified",
        category_label=taxonomy.label_for(category) if category else "Unclassified incident",
        role=role,
        severity=pb.severity,
        title=pb.title,
        problem=problem,
        likely_causes=list(pb.likely_causes),
        immediate_actions=[_fill(s, service) for s in pb.immediate],
        investigate=[CommandHint(label=lbl, command=_fill(cmd, service)) for lbl, cmd in pb.investigate],
        permanent_fix=list(pb.fix),
        verify=list(pb.verify),
        prevent=list(pb.prevent),
        rollback=_fill(pb.rollback, service) or None,
        caution=pb.caution or None,
        matched_signals=signals,
        confidence_pct=round(conf, 1),
        proven_in_history=_proven_fixes(hits, category) if role == "primary" else [],
    )


def build_recommendations(profile: taxonomy.Profile, service: str,
                          hits: Optional[list[MemoryHit]] = None) -> list[Recommendation]:
    """Primary recommendation for the incident's category (+ a secondary one when
    the text strongly matches a second category). Always returns at least one."""
    hits = hits or []
    primary = profile.primary
    recs = [_build_one(playbook_for(primary), primary, profile, service, "primary", hits)]
    second = profile.secondary
    if second and second != primary:
        recs.append(_build_one(playbook_for(second), second, profile, service, "secondary", hits))
    return recs


def steps_from_recommendations(recs: list[Recommendation], limit: int = 8) -> list[str]:
    """Flat ordered action list (used when the LLM produced none / failed)."""
    if not recs:
        return []
    r = recs[0]
    steps = list(r.immediate_actions) + list(r.permanent_fix[:2]) + list(r.verify[:1])
    return steps[:limit]
