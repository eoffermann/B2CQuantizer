# Manual end-to-end smoke procedure

This is a **manual** procedure, not an automated test suite. It exercises the full B2CQuantizer
pipeline end-to-end against a real (small) model on real hardware — something automated unit tests
cannot do, since the app's job is orchestrating GPU-bound and CPU-bound subprocess toolchains
(`llm-compressor`, `llama.cpp`) against real model weights.

Run this:
- After any change to the Docker image (Dockerfile, `scripts/build_llama_cpp.sh`, the mmproj
  patch script) — image-level changes can't be validated by unit tests alone.
- Before trusting a freshly-deployed pod with a real (large) quantization job.
- After fixing a bug reported in [`docs/troubleshooting.md`](../../docs/troubleshooting.md),
  particularly the mmproj patch issue, to confirm the fix actually applies at runtime.

For automated GPU-dependent tests that run as part of the normal test suite (opt-in, smaller in
scope than this procedure), see `tests/integration/` — set `GPU_INTEGRATION=1` to enable them.
This smoke procedure is the broader, real-world complement to those.

## Prerequisites

- A pushed B2CQuantizer image (see [`docs/deployment.md` §3](../../docs/deployment.md#3-publishing-the-docker-image)).
- A HuggingFace account/token with write access to your own namespace (to create the output repos)
  and, for step 5, write access to the source model repo if you want to exercise the README-update
  path against a repo you own — a public model you don't own is fine for steps 1–4 and 6, but step 5
  will correctly log a skip-with-warning rather than a failure in that case (`SPEC.md` §9).
- `curl`, `vllm` (or access to a vLLM install), and `llama-cli` available somewhere you can run
  them to validate outputs (step 6) — these don't need to be on the pod itself if you'd rather
  download the small outputs locally.

## Procedure

### 1. Deploy the image to a RunPod H100 80GB pod

Follow [`docs/deployment.md`](../../docs/deployment.md) end to end: push the image, create a RunPod
pod with an H100 80GB GPU, expose HTTP port 8000, and confirm you can reach the UI through RunPod's
proxy/tunnel.

An H100 80GB pod is more than sufficient for the small model used in this procedure (step 2) — it's
specified here for consistency and because it's a supported tier for the full 24B-model catalog if
you want to extend the smoke test later. **Do not** select NVFP4 in this procedure on an H100 — it's
Blackwell-only (`SPEC.md` §7) and will fail or be grayed out by design; that's expected, not a bug,
on this GPU tier.

Sanity-check the pod is healthy before proceeding:
```bash
curl -sSf https://<your-pod-proxy-url>/health
```
Expected: `{"status":"ok"}`.

### 2. Point at a public small Mistral model

Use a small, public, non-gated Mistral model so the smoke run is fast and doesn't require accepting
a license gate. `mistralai/Mistral-7B-v0.1` is a good default — public, plain-text (not multimodal),
Mistral-architecture.

In the setup page (`GET /`):
- **Source model:** `mistralai/Mistral-7B-v0.1`
- **HF token:** your token (write access to your own namespace, at minimum).
- **Upload owner:** your namespace (should auto-populate from the token via `whoami` once entered).

Because this model is text-only, the mmproj checkbox should **not** appear (it's only shown for
detected-multimodal sources, per `SPEC.md` §10) — that's expected for this model, not a defect.

### 3. Select Q4_K_M + Q8_0 + W4A16_GPTQ, no calibration override

- **Quant selection:** check exactly three formats:
  - `Q4_K_M` (GGUF K-quant — Lane B, no calibration needed)
  - `Q8_0` (GGUF misc — Lane B, no calibration needed)
  - `W4A16_GPTQ` (safetensors — Lane A, needs calibration)
- **Calibration source:** leave on the default, **Bundled generic corpus** — do not upload a JSONL
  or enter an HF dataset ID. This exercises the zero-friction path (`SPEC.md` §8) and keeps the
  smoke run self-contained.
- Leave **Update source README** at its default (ON) — this is what step 5 verifies.
- Review the repo-name preview table (should show `<owner>/Mistral-7B-v0.1-Q4_K_M` /
  `-Q8_0` sharing a single `<owner>/Mistral-7B-v0.1-GGUF` repo, and
  `<owner>/Mistral-7B-v0.1-W4A16_GPTQ` as its own repo) and Launch.

This selection deliberately exercises both lanes concurrently (Lane A: one safetensors format;
Lane B: two GGUF formats sharing one BF16 GGUF intermediate) without pulling in the imatrix
pre-step (no I-quant selected) — a smaller, faster smoke surface than the full catalog while still
covering both lanes, the calibration path, and both upload shapes (folder upload vs. single-file
upload).

### 4. Verify all three uploads land in the expected repo layout

Watch the job view (`GET /jobs/<job_id>`) until all three quants show `DONE`. Then confirm on the
HF Hub (web UI or `huggingface-cli`/`huggingface_hub`):

- `<owner>/Mistral-7B-v0.1-GGUF` exists and contains two files: one for `Q4_K_M`, one for `Q8_0`
  (per `SPEC.md` §9, all GGUFs for a source model share a single repo, one file each).
- `<owner>/Mistral-7B-v0.1-W4A16_GPTQ` exists as its own repo containing the full safetensors
  folder output (config, tokenizer files, quantized weights) — per `SPEC.md` §9, each safetensors
  format gets its own dedicated repo.
- The public/private setting you chose at launch matches what was actually created on the Hub.
- Each quant's row in the job view shows a non-error `upload_url`/link and a `DONE` status with a
  plausible output size.

### 5. Verify source repo README has a `## Quantizations` section

If your token has write access to the source repo (i.e., you own `mistralai/Mistral-7B-v0.1` — for
most operators running this smoke test you will **not**, since it's a third-party base model):

- Check the job's final report block for the README-update outcome per `SPEC.md` §10/§11 — expect
  a **logged warning, not a failure**, since you don't own the source repo (403 on the write
  attempt). Confirm the job itself still completed successfully and quants stayed uploaded — this
  is the expected/correct behavior (`SPEC.md` §11, "README commit fails" row: "quants stay
  uploaded").

If you substitute a source model you **do** own for this step (recommended at least once to
exercise the write path fully):
- Confirm the source repo's README now has a `## Quantizations` section formatted as a Markdown
  table (`Format | Repo | Notes`) listing all three quants with working links to their repos
  (`SPEC.md` §9).
- Re-run the same job (or a new job against the same source model) and confirm the section is
  **replaced**, not duplicated — the update is specified as idempotent (`SPEC.md` §9).

### 6. Verify each quant is loadable

Confirm the outputs aren't just present on the Hub but actually load in their target runtime.

**vLLM smoke** (safetensors quant):
```bash
MODEL_NAME=<owner>/Mistral-7B-v0.1-W4A16_GPTQ vllm serve "$MODEL_NAME" --max-model-len 4096
```
Expected: vLLM loads the model and starts serving without erroring on the quantized weights/config
(GPTQ Marlin kernel path per `SPEC.md` §7 notes). A successful health check / one completion
request against the running server is sufficient — you don't need a full benchmark.

**llama.cpp smoke** (GGUF quants):
```bash
# Download the two GGUF files from <owner>/Mistral-7B-v0.1-GGUF first, then:
llama-cli -m Mistral-7B-v0.1-Q4_K_M.gguf -p "Hello" -n 16
llama-cli -m Mistral-7B-v0.1-Q8_0.gguf -p "Hello" -n 16
```
Expected: both GGUF files load without errors and produce coherent (if brief) text output — this
confirms the K-quant conversion and upload round-tripped correctly.

## Pass/fail

The smoke procedure **passes** when:
- All three quants reach `DONE` with valid upload URLs (step 4).
- The README-update outcome matches the expected behavior for whether you own the source repo
  (step 5) — either a populated `## Quantizations` section, or a clean warning-and-skip.
- Both the vLLM and llama.cpp loads in step 6 succeed.

Any deviation — a quant stuck in an unexpected state, a missing repo, a README section that didn't
render or didn't idempotently replace, or a load failure in step 6 — is a regression. Check
[`docs/troubleshooting.md`](../../docs/troubleshooting.md) first; if the symptom doesn't match a
documented failure mode, treat it as a new bug to investigate against the relevant lane/module in
[`PLAN.md`](../../PLAN.md).

## See also

- [`docs/deployment.md`](../../docs/deployment.md) — pod setup this procedure assumes is already done.
- [`docs/troubleshooting.md`](../../docs/troubleshooting.md) — common failure modes if a step above
  doesn't behave as expected.
- [`SPEC.md`](../../SPEC.md) — authoritative design spec.
