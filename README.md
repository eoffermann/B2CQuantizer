# B2CQuantizer

**Self-service batch quantization for Mistral / Mistral3 models, in a browser, on a rented GPU.**

B2CQuantizer is a Docker web app designed to run on a [RunPod](https://runpod.io) GPU pod. Point it at a HuggingFace Mistral or Mistral3 model, select any combination of ~30 quantization formats, enter your HF token, and hit Launch. It downloads the source model once, quantizes it across two concurrent pipelines, uploads every result to its own HuggingFace repo, and appends a `## Quantizations` section with links to the source model's README. Rent a big GPU for an afternoon, drive it from a browser, walk away, come back to a published quant pack.

## Why

Quantizing 24B-parameter models (Mistral-Small-3.2-24B and similar) locally on a 48 GB card is tight to impossible: BF16 weights alone are ~48 GB, and GPTQ/NVFP4 calibration adds per-layer Hessians or activation statistics on top. Shell-script pipelines that chain quants sequentially are hard-coded per run, abort the whole chain on any single failure, and give no visibility into progress. B2CQuantizer replaces that with a disposable, self-service pod: one job, many formats, per-quant failure isolation, and live progress in the browser.

## What it produces

A single job can emit any selection of **30 output formats**:

| Group | Formats | Notes |
|---|---|---|
| GGUF K-quants (9) | Q2_K, Q3_K_S/M/L, Q4_K_S/M, Q5_K_S/M, Q6_K | `llama-quantize`, no calibration |
| GGUF I-quants (12) | IQ1_S/M, IQ2_XXS/XS/S/M, IQ3_XXS/XS/S/M, IQ4_XS/NL | Require an imatrix pre-step (computed once per job from calibration data) |
| GGUF misc (3) | Q8_0, F16, BF16 | Repacks / reference quality |
| GGUF multimodal (1) | mmproj-f16 | Auto-included for Mistral3 multimodal models; includes runtime patches for two upstream llama.cpp pixtral-export bugs |
| safetensors (5) | W4A16 GPTQ, W4A16 AWQ, NVFP4, FP8 E4M3, FP8 E5M2 | Via `llm-compressor`; served by vLLM. NVFP4 requires a Blackwell GPU |

**Repo layout on upload:** each safetensors format gets its own repo (`<owner>/<model>-GPTQ`, `-AWQ`, `-NVFP4`, …); all GGUF files share a single `<owner>/<model>-GGUF` repo. Repo names are editable in the UI before launch, with a public/private toggle.

## How it works

After a one-time source download, the job runner drives two lanes concurrently:

- **Lane A (GPU-bound, safetensors):** loads the BF16 source onto the GPU once, then runs each selected `llm-compressor` recipe serially against the resident model — W4A16 GPTQ (symmetric, g128), W4A16 AWQ (asymmetric, g128, with explicit per-decoder-layer scoped mappings for multimodal Mistral3), NVFP4, FP8. Each result is uploaded and deleted locally before the next format starts.
- **Lane B (CPU-bound, GGUF):** converts the source safetensors to a canonical BF16 GGUF once, computes an imatrix once if any I-quant is selected, then runs `llama-quantize` per format. For multimodal Mistral3 sources it also exports the vision projector (`mmproj-f16`) after patching llama.cpp's converter, with a hard assertion that the `v.token_embd.img_break` tensor made it into the output.

Failure isolation is per-quant: a failed format is marked FAILED with its error and the lane moves on. A lane-setup failure (GPU OOM, GGUF conversion failure) skips that lane's remaining quants while the other lane continues. Both lanes stream stdout/stderr to the browser over Server-Sent Events.

### Calibration

One calibration source per job, feeding both lanes uniformly (llm-compressor calibration and llama-imatrix see the same distribution):

1. **Bundled generic corpus** — ~1000 samples (wikitext + chat-style) baked into the image; zero friction.
2. **Uploaded JSONL** — OpenAI-messages format.
3. **HF dataset ID** — downloaded with the job's token; falls back to the bundled corpus if unreachable.

For the safetensors lane, calibration rendering is automatic and artifact-class-aware: if the source model repo carries a `tekken.json` (all FrndoBrain-class merges do, byte-identical to the base Mistral repo), calibration is rendered exclusively through `mistral-common` with the tools-block flipped to the first user turn, matching the training/serving invariant — never through the HF chat template, since even a byte-level rendering difference would bake calibration/inference divergence into the quantized weights. A calibration sample that fails to render under mistral-common hard-fails the job rather than silently falling back, since that means the calibration corpus doesn't match the training contract. Generic Mistral models (no `tekken.json`) calibrate via the standard HF `apply_chat_template` path. See `SPEC.md` §14 for the full invariant set and version pin.

### Source README update

After uploads complete, the app commits a canonical `## Quantizations` table (Format | Repo | Notes) to the source repo's `main` branch — idempotent, replacing the section on subsequent runs. If the token can't write to the source repo (e.g., quantizing someone else's base model), it logs a warning and skips; quants stay uploaded.

## Quick start

1. Launch a RunPod pod:
   - **GPU:** H100 80 GB or A100 96 GB for 24B models. RTX 5090 / RTX PRO 6000 (Blackwell) required if you select NVFP4 — the UI grays it out otherwise.
   - **Disk:** ≥200 GB for 24B models (the app refuses to launch a job with <150 GB free).
2. Run the image, exposing port 8000:
   ```bash
   docker run -d --gpus all -p 8000:8000 ghcr.io/bigblueceiling/b2cquantizer:latest
   ```
3. Open the UI via RunPod's port forwarding (SSH tunnel or access URL) — **the app has no auth or TLS of its own; the pod's port exposure is the trust boundary.**
4. Enter the source model repo ID and your HF token, pick a calibration source, check the formats you want, review the generated repo names, and Launch.
5. Watch the live dashboard: per-quant status table, elapsed times, and streaming log panes for both lanes. A final report lists successes, failures with reasons, and the README-update outcome.

## Security model

- The HF token lives only in the FastAPI process's memory — never written to disk, never logged, never exported to subprocess environments — and is wiped when the job ends (success or failure). Each new job requires re-entering it.
- The image ships no credentials; uploaded calibration files live on the pod's ephemeral disk and are deleted at job end.
- No in-app authentication: run it only behind RunPod's tunnel/proxy.

## Scope and limitations (v1)

- **Mistral / Mistral3 only.** Other architectures (Llama, Qwen, Phi, Gemma, …) are refused at launch with a clear error — the multimodal handling, scoped AWQ mappings, and mmproj export are Mistral3-specific.
- Single user, single job at a time. No multi-tenancy, no auth.
- Everything is in-memory: a pod restart loses in-progress jobs and session history (already-uploaded artifacts stay uploaded).
- No automatic retries of failed quants (uploads retry with backoff; quants don't) and no automatic pod provisioning.
- No EXL2 (separate toolchain).

Deferred to v2: broader architectures, EXL2, batch submission, job persistence, retry-in-place, auto-provisioning.

## Tech stack

FastAPI (Python 3.10) + Uvicorn backend · HTMX + Alpine.js + Tailwind (CDN) frontend, no SPA framework or Node.js · Server-Sent Events for live progress · `llm-compressor` for safetensors quants · `llama.cpp` (built from source in-image, CUDA-enabled) for GGUF conversion, imatrix, and quantization · `transformers` + `accelerate` for model loading · `huggingface_hub` for downloads/uploads · Docker base `nvidia/cuda:12.8.0-devel-ubuntu22.04` (CUDA 12.8 for Blackwell SM120 / NVFP4 support).

## Project documents

- [`SPEC.md`](SPEC.md) — the approved design specification (pipeline model, failure modes, UI surface, security).
- [`PLAN.md`](PLAN.md) — the task-by-task implementation plan.
- `docs/deployment.md` — RunPod deployment guide.
- `docs/troubleshooting.md` — common failure modes and fixes.

## License

MIT
