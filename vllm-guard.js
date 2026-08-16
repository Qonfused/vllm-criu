#!/usr/bin/env bun
// vllm-guard — reverse proxy + auto sleep/wake supervisor for a vLLM server.
//
// Sits in front of the vLLM OpenAI API server:
//   * Forwards all API traffic (streaming-safe; SSE passes through unbuffered)
//   * Wakes the engine on demand when a request arrives while it is asleep
//   * Puts the engine to sleep after IDLE_TIMEOUT seconds of no activity,
//     releasing GPU memory (weights offloaded to CPU RAM, KV cache discarded)
//   * Optionally asks the vLLM launcher to checkpoint the post-level-2 process
//     state to disk (CHECKPOINT_MODE=criu)
//   * Blocks vLLM's dev-mode endpoints so they stay on the internal network
//
// If the upstream was started without --enable-sleep-mode (GET /is_sleeping
// -> 404), the guard degrades to a plain reverse proxy and logs it once.
//
// Config (env):
//   VLLM_URL       upstream base URL          (default http://vllm:8001)
//   LISTEN_PORT    port to bind               (default 8001)
//   IDLE_TIMEOUT   seconds of inactivity before sleep (default 300)
//   POLL_INTERVAL  seconds between /metrics polls     (default 15)
//   WAKE_TIMEOUT   max seconds to wait for a wake-up  (default 120)
//   SLEEP_LEVEL    1 = weights to RAM, 2 = discard weights (default 2)
//   CONTROL_URL    internal vLLM launcher URL        (default http://vllm:9000)
//   CHECKPOINT_MODE off or criu; criu is opt-in       (default off)

const VLLM = (process.env.VLLM_URL ?? "http://vllm:8001").replace(/\/+$/, "");
const CONTROL = (process.env.CONTROL_URL ?? "http://vllm:9000").replace(/\/+$/, "");
const PORT = Number(process.env.LISTEN_PORT ?? 8001);
const IDLE_TIMEOUT_MS = Number(process.env.IDLE_TIMEOUT ?? 300) * 1000;
const POLL_INTERVAL_MS = Number(process.env.POLL_INTERVAL ?? 15) * 1000;
const WAKE_TIMEOUT_MS = Number(process.env.WAKE_TIMEOUT ?? 120) * 1000;
const SLEEP_LEVEL = process.env.SLEEP_LEVEL ?? "2";
const CHECKPOINT_MODE = process.env.CHECKPOINT_MODE ?? "off";

if (!new Set(["1", "2"]).has(SLEEP_LEVEL)) {
  throw new Error(`SLEEP_LEVEL must be 1 or 2, got ${SLEEP_LEVEL}`);
}
if (!new Set(["off", "criu"]).has(CHECKPOINT_MODE)) {
  throw new Error(`CHECKPOINT_MODE must be off or criu, got ${CHECKPOINT_MODE}`);
}

// vLLM dev-mode endpoints (exposed when VLLM_SERVER_DEV_MODE=1). The docs say
// these "should not be exposed to users" — /collective_rpc in particular
// allows arbitrary worker RPCs. Never proxy them; the guard itself calls
// them directly on the internal network.
const BLOCKED_PATHS = new Set([
  "/sleep",
  "/wake_up",
  "/collective_rpc",
  "/is_sleeping",
  "/server_info",
  "/rlhf",
]);

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const log = (...a) => console.log(new Date().toISOString(), ...a);

let inFlight = 0; // requests currently being proxied
let lastActive = Date.now();
let lastGen = null; // last observed vllm:generation_tokens_total
let sleepMode = null; // null = unprobed, true = available, false = unavailable
let checkpointed = false;
let wakePromise = null; // single-flight wake-up
let sleepPromise = null; // single-flight sleep

async function upstream(path, opts = {}) {
  return fetch(VLLM + path, { redirect: "manual", ...opts });
}

function parseSleepState(body) {
  try {
    const j = JSON.parse(body);
    if (typeof j.is_sleeping === "boolean") return j.is_sleeping;
  } catch {}
  return body.trim().toLowerCase() === "true";
}

async function isSleeping() {
  const r = await upstream("/is_sleeping");
  if (r.status === 404) {
    if (sleepMode !== false) log("upstream has no sleep mode — running as plain proxy");
    sleepMode = false;
    return false;
  }
  if (!r.ok) throw new Error(`/is_sleeping -> ${r.status}`);
  sleepMode = true;
  return parseSleepState(await r.text());
}

async function postUpstream(path, init = {}) {
  const r = await upstream(path, { method: "POST", ...init });
  if (!r.ok) {
    throw new Error(`${path} -> ${r.status}: ${await r.text().catch(() => "")}`);
  }
  return r;
}

async function postControl(path, init = {}) {
  const r = await fetch(CONTROL + path, { method: "POST", redirect: "manual", ...init });
  const body = await r.text().catch(() => "");
  if (!r.ok) throw new Error(`${path} -> ${r.status}: ${body}`);
  try {
    return JSON.parse(body);
  } catch {
    return { ok: true };
  }
}

// The guard's checkpointed flag is intentionally in-memory. The vLLM
// container can nevertheless be recreated independently of the guard, so a
// previously checkpointed flag must not make us restore an image belonging to
// an older launcher process tree. The launcher state is authoritative when it
// is available: a fresh launcher in `starting` or `running` state wins over
// the guard's stale flag.
async function reconcileCheckpointState() {
  if (!checkpointed) return;
  try {
    const r = await fetch(CONTROL + "/launcher/healthz", { redirect: "manual" });
    const body = await r.json();
    if (["starting", "running"].includes(body.state)) {
      checkpointed = false;
      log(`launcher reports state=${body.state}; ignoring stale checkpoint flag`);
    }
  } catch (e) {
    log(`could not reconcile launcher checkpoint state: ${e.message}`);
  }
}

async function reloadWeights() {
  const r = await postUpstream("/collective_rpc", {
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      method: "reload_weights",
      timeout: Math.ceil(WAKE_TIMEOUT_MS / 1000),
    }),
  });
  await r.text().catch(() => "");
}

// Ensure the engine can serve a request right now. Concurrent callers share
// one wake-up (single-flight). Resolves once the engine is fully awake.
async function ensureAwake() {
  if (sleepMode === false) return;
  if (wakePromise) return wakePromise;
  wakePromise = (async () => {
    try {
      const t0 = Date.now();
      if (checkpointed) {
        await reconcileCheckpointState();
      }
      if (checkpointed) {
        log("engine is checkpointed — restoring launcher process tree");
        await postControl("/launcher/resume");
        checkpointed = false;
      }
      if (!(await isSleeping())) return;
      log("engine is asleep — waking up before forwarding request");
      if (SLEEP_LEVEL === "2") {
        log("level 2 wake: reallocating model weights");
        await postUpstream("/wake_up?tags=weights");
        log("level 2 wake: reloading model weights from the model source");
        await reloadWeights();
        log("level 2 wake: reallocating KV cache");
        await postUpstream("/wake_up?tags=kv_cache");
      } else {
        await postUpstream("/wake_up");
      }
      const deadline = Date.now() + WAKE_TIMEOUT_MS;
      for (;;) {
        if (!(await isSleeping())) break;
        if (Date.now() > deadline) throw new Error(`wake-up timed out after ${WAKE_TIMEOUT_MS / 1000}s`);
        await sleep(250);
      }
      log(`engine awake in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
    } finally {
      wakePromise = null;
    }
  })();
  return wakePromise;
}

// Sum a Prometheus counter/gauge across all label sets.
function sumMetric(text, name) {
  const re = new RegExp(
    `^${name}(\\{[^}]*\\})?\\s+(-?[0-9]+(\\.[0-9]+)?([eE][-+]?[0-9]+)?)$`,
    "gm",
  );
  let sum = 0,
    found = false,
    m;
  while ((m = re.exec(text))) {
    sum += Number(m[2]);
    found = true;
  }
  return found ? sum : null;
}

async function trySleep() {
  // A newly started guard has not probed /is_sleeping yet. Allow the first
  // idle check to discover the upstream state; returning while sleepMode is
  // null would permanently disable automatic sleep for an idle fresh server.
  if (sleepMode === false || sleepPromise || inFlight > 0) return;
  let currentlySleeping;
  try {
    currentlySleeping = await isSleeping();
  } catch (e) {
    log(`pre-sleep check failed: ${e.message}`);
    return;
  }
  if (currentlySleeping) return;
  sleepPromise = (async () => {
    const t0 = Date.now();
    log(`idle for ${IDLE_TIMEOUT_MS / 1000}s — sleeping engine (level ${SLEEP_LEVEL}) to release GPU memory`);
    if (CHECKPOINT_MODE === "criu" && SLEEP_LEVEL === "2") {
      try {
        await postControl("/launcher/suspend");
        checkpointed = true;
        sleepMode = true;
        log(`engine checkpointed to disk in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
        return;
      } catch (e) {
        // CRIU mode is specifically selected to remove the vLLM CUDA process
        // and its residual GPU allocations. Falling back to in-process sleep
        // would violate that contract, so leave the engine running and retry
        // on the next idle-loop iteration.
        log(`checkpoint failed; not falling back to level ${SLEEP_LEVEL} sleep: ${e.message}`);
        return;
      }
    }
    await postUpstream(`/sleep?level=${SLEEP_LEVEL}`);
    log(`engine asleep in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  })().finally(() => {
    sleepPromise = null;
  });
  try {
    await sleepPromise;
  } catch (e) {
    log(`sleep failed: ${e.message}`);
  }
}

async function idleLoop() {
  for (;;) {
    await sleep(POLL_INTERVAL_MS);
    if (checkpointed) continue;
    if (inFlight > 0) {
      lastActive = Date.now();
      continue;
    }
    let text;
    try {
      const r = await upstream("/metrics");
      if (!r.ok) throw new Error(`status ${r.status}`);
      text = await r.text();
    } catch (e) {
      log(`metrics poll failed: ${e.message}`);
      continue;
    }
    const gen = sumMetric(text, "vllm:generation_tokens_total");
    const running = sumMetric(text, "vllm:num_requests_running") ?? 0;
    const waiting = sumMetric(text, "vllm:num_requests_waiting") ?? 0;
    const active =
      (gen != null && lastGen != null && gen !== lastGen) || running > 0 || waiting > 0;
    if (gen != null) lastGen = gen;
    if (active) {
      lastActive = Date.now();
      continue;
    }
    if (Date.now() - lastActive >= IDLE_TIMEOUT_MS) await trySleep();
  }
}

Bun.serve({
  port: PORT,
  idleTimeout: 0, // long non-streaming completions hold the connection with no bytes
  async fetch(req) {
    const url = new URL(req.url);

    if (url.pathname === "/healthz") {
      return Response.json({
        ok: true,
        sleepMode,
        checkpointed,
        checkpointMode: CHECKPOINT_MODE,
        inFlight,
        idleSeconds: Math.round((Date.now() - lastActive) / 1000),
      });
    }

    if (BLOCKED_PATHS.has(url.pathname)) {
      return new Response("forbidden: vLLM dev endpoint", { status: 403 });
    }

    inFlight++;
    lastActive = Date.now();
    try {
      await ensureAwake();
      const target = `${VLLM}${url.pathname}${url.search}`;
      const headers = new Headers(req.headers);
      headers.set("host", new URL(VLLM).host);
      const res = await fetch(target, {
        method: req.method,
        headers,
        body: req.method === "GET" || req.method === "HEAD" ? undefined : req.body,
        redirect: "manual",
        signal: req.signal,
      });
      return new Response(res.body, {
        status: res.status,
        statusText: res.statusText,
        headers: res.headers,
      });
    } catch (e) {
      return new Response(`upstream error: ${e.message}`, { status: 502 });
    } finally {
      inFlight--;
    }
  },
});

log(
  `vllm-guard listening on :${PORT} -> ${VLLM} ` +
  `(idle timeout ${IDLE_TIMEOUT_MS / 1000}s, poll ${POLL_INTERVAL_MS / 1000}s, ` +
  `sleep level ${SLEEP_LEVEL}, checkpoint mode ${CHECKPOINT_MODE})`,
);
idleLoop();
