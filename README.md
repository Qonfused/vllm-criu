# vllm-criu

`vllm-criu` saves and restores a running vLLM process tree with NVIDIA's [`cuda-checkpoint`](https://github.com/NVIDIA/cuda-checkpoint) utility and [CRIU](https://criu.org/). This repository includes a Python launcher, an idle proxy implemented in `vllm-guard.js`, and EngineCore patches for restoring the service.

The [vLLM CUDA checkpoint/restore RFC](https://github.com/vllm-project/vllm/issues/34303) provides a breakdown of the CUDA and CRIU mechanisms that this project implements. Our launcher first calls vLLM level-2 sleep, then invokes `cuda-checkpoint` and CRIU around the running process. During restore, EngineCore runtime patches reconnect the vLLM process and rebuild the model weights and KV cache.

## Suspend and restore sequence

After the configured idle period, `vllm-guard.js` sends `POST /launcher/suspend` to the launcher. The launcher then calls vLLM's `/sleep?level=2` endpoint, prepares the worker, locks and checkpoints the CUDA process, and runs `criu dump` on the vLLM process tree. Afterwards, CRIU writes the images to the configured checkpoint directory.

On resume, `POST /launcher/resume` first runs `criu restore` and restores the CUDA process. Once the API server is ready, the EngineCore patches replace stale socket readers and reconnect the API and EngineCore channels. The worker then reinitializes its CUDA state, reloads model weights, allocates the KV cache, and resumes scheduling. Graph preservation is optional. When it is disabled or cannot reuse the restored graph state, the worker recaptures graphs or uses eager execution.

Because CRIU restores the process tree, a successful resume does not start a new vLLM process. However, the worker still transfers model weights to the GPU and rebuilds the KV cache, so those costs remain part of resume time. After a successful restore, the launcher deletes the checkpoint directory. The next suspend creates a new checkpoint. If restore fails and fallback is enabled, the launcher starts a fresh vLLM process.

## Requirements

The package needs Linux, a CUDA-aware CRIU installation, the [`cuda-checkpoint`](https://github.com/NVIDIA/cuda-checkpoint) utility, and a vLLM image containing these runtime hooks. The driver, CUDA runtime, CRIU support, model configuration, and checkpoint environment must remain compatible between suspend and restore. Unit tests run without a GPU or a live service.

## Development and testing

Use [UV](https://docs.astral.sh/uv/) to install the development environment and run the repository's unit tests:

```bash
uv sync
uv run pytest -q
```

By default, live CRIU integration tests are skipped as it suspends and restores a live service. To run it, first make the launcher, vLLM server, and guard available, then set the `VLLM_CRIU_INTEGRATION` environment variable to `1` and run pytest with the `criu_integration` marker:

```bash
VLLM_CRIU_INTEGRATION=1 uv run pytest -q -m criu_integration -s
```

During the integration test, the suite calls the live suspend and restore endpoints, checks the CUDA-graph inventory before and after restore, and sends requests through the guard. Presently it's TP=1 scope does not test multi-GPU or NCCL graph behavior.

## Repository layout

```text
src/vllm_criu/             Python package and launcher
src/vllm_criu/enginecore/  EngineCore lifecycle and restore hooks
tests/                     Unit tests and the opt-in CRIU test
vllm-guard.js              Idle proxy and suspend/restore controller
launcher.py                Container entrypoint compatibility shim
```
