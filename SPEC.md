# B2CQuantizer — Design Specification

**Status:** approved 2026-07-25.
**Purpose:** RunPod-hosted Docker app that quantizes a HuggingFace Mistral / Mistral3 model into a user-selected set of GGUF + safetensors formats, uploads each result to its own HuggingFace repo, and appends a `## Quantizations` section to the source model's README with links to every generated repo.

---

## 1. Problem being solved

Producing quants of FrndoBrain-class models (Mistral-Small-3.2-24B and similar) locally on a 48 GB A6000 is tight-to-impossible for BF16 → W4A16 GPTQ / NVFP4 calibration (weights alone are ~48 GB; add per-layer Hessians or activation stats for calibration and you're over the card). The current one-off `chain_quant_run_004.sh` orchestrates W4A16 → NVFP4 → AWQ sequentially but is hard-coded per run, aborts the chain on any single failure, and has no UI.

The desired shape is a self-service pod: rent a big GPU on RunPod for the duration of a batch of quants, drive it from a browser, walk away, come back to a set of HF repos + an updated source README.

## 2. Users and use cases

- **Single user** (the operator running the pod), single job at a time. No multi-tenancy.
- **Primary use case:** operator has just trained (or is about to release) a new FrndoBrain fine-tune. They want to ship a GGUF pack for LM Studio + one or more safetensors quants for vLLM on RunPod backend + updated README on the source model.
- **Secondary use case:** operator wants to quant a third-party Mistral base model for their own experimentation. In this case they own the destination quant repos but not the source; the README-update step logs a warning and skips.

## 3. Non-goals (v1)

- Multi-user / authenticated web UI. The pod's HTTP port is behind RunPod's SSH tunnel or their access-URL proxy; no in-app auth.
- Non-Mistral architectures (Llama, Qwen, Phi, Gemma, …). The multimodal handling, scoped AWQ mappings, and mmproj export are Mistral3-specific. Broader architecture support is a v2 concern.
- EXL2 quants (separate toolchain).
- Automatic RunPod pod provisioning. Operator manages the pod lifecycle themselves.
- Job persistence across pod restarts. Everything in-memory; a restarted pod is a blank slate.
- Automatic retry of failed quants. Manual retry via a new job.

## 4. Deployment shape

- Docker image, published somewhere the operator can pull (Docker Hub, GHCR, or private registry).
- Operator creates a RunPod pod with a GPU appropriate to the workload — 80 GB H100 or 96 GB A100 for 24B models. Blackwell tier (RTX 5090 / RTX PRO 6000 Blackwell) required if NVFP4 is in the quant selection.
- Pod is expected to expose port 8000 (the app's HTTP port) via RunPod's SSH tunnel or their public access URL. The app itself does no TLS termination and no authentication.
- Ephemeral pod disk holds the source download and in-progress artifacts. Recommended: **≥200 GB** for 24B-parameter models. Aggressive cleanup deletes each artifact after successful upload.
- The container runs one FastAPI process on startup and stays up until the operator destroys the pod. Multiple jobs can run sequentially in a single pod session; each new job requires re-entering the HF token.

## 5. Tech stack

- **Backend:** FastAPI (Python 3.10). Async event loop hosts one job runner at a time.
- **Frontend:** FastAPI-served HTML + HTMX + minimal Alpine.js. No SPA framework, no Node.js in the image. Tailwind loaded via CDN for styling.
- **Live progress:** Server-Sent Events (SSE). One-way stream is sufficient; simpler than WebSockets.
- **Quant toolchain:**
  - `llm-compressor` (safetensors): W4A16 GPTQ, W4A16 AWQ, NVFP4, FP8 E4M3, FP8 E5M2.
  - `llama.cpp` built from source in-image: `convert_hf_to_gguf.py`, `llama-imatrix`, `llama-quantize`.
  - `transformers` + `accelerate` for model loading.
  - `huggingface_hub` for downloads and uploads.
- **Docker base:** `nvidia/cuda:12.8.0-devel-ubuntu22.04` (CUDA 12.8 = Blackwell SM120 support for NVFP4; matches the `frndobrain-cuda128` image family the operator already runs).

## 6. Pipeline model

After a one-time source-model download, the job runner starts two lanes concurrently:

### Lane A (GPU-bound): safetensors quants

1. Load source BF16 into GPU via `AutoModelForImageTextToText` (multimodal detected from `config.json`).
2. For each selected safetensors quant in the recipe list — serially, reusing the loaded model in GPU memory:
   - Build the appropriate `llm-compressor` recipe (GPTQModifier for W4A16, AWQModifier for AWQ, QuantizationModifier for NVFP4 / FP8).
   - Ignore patterns exclude vision tower, multi-modal projector, and `lm_head`.
   - For AWQ on multimodal Mistral3: use explicit per-decoder-layer scoped AWQ mappings (default `re:.*input_layernorm$` etc. globs all 40 decoder layernorms into one set and errors out).
   - Run the calibration pass with the operator's chosen calibration data.
   - Save the quantized model to a local directory.
   - Upload the directory to `<owner>/<sourcemodel>-<FORMAT>` as its own HF repo.
   - Delete the local directory. Next quant reuses the still-loaded GPU model.
3. Lane completion: log summary, mark lane done.

### Lane B (CPU-bound): GGUF quants

1. Read the source safetensors from disk (already downloaded). Run `convert_hf_to_gguf.py --outtype bf16` to produce a canonical BF16 GGUF (~45 GB for a 24B model). This is the shared substrate for every downstream GGUF quant.
2. **If any I-quant is selected in the recipe** — one-time step per job: run `llama-imatrix` against the BF16 GGUF using calibration text (imatrix quants need this or their quality is significantly degraded). This produces a `.imatrix` file used by all I-quants.
3. For each selected GGUF quant — serially:
   - Run `llama-quantize <bf16.gguf> <out.gguf> <QUANT>` (or `--imatrix <imatrix.file>` for I-quants).
   - Upload the resulting `.gguf` file to the shared `<owner>/<sourcemodel>-GGUF` repo.
   - Delete the local file.
4. If the source model is multimodal (Mistral3ForConditionalGeneration detected): after running the LM GGUFs, apply the local `patch_convert_hf_to_gguf.py` fixes to the llama.cpp copy inside the image (see the operator's CLAUDE.md gotcha #9 in the unsloth repo — Mistral3 pixtral mmproj export has two upstream bugs that need patching), then run `convert_hf_to_gguf.py --mmproj` and upload the resulting `-mmproj-f16.gguf` file to the same `-GGUF` repo. Verify `v.token_embd.img_break` is present in the mmproj output as a hard assertion.
5. Delete the BF16 GGUF intermediate once all quants are done. Lane complete.

### Cross-lane coordination

- Both lanes stream their stdout/stderr to a per-quant progress bus that the SSE endpoint pulls from.
- Per-quant failure is caught at the quant boundary: mark that quant FAILED with the error, continue to the next quant in the lane.
- Per-lane setup failure (e.g., GPU load OOM in Lane A, BF16 GGUF conversion failure in Lane B) marks all downstream quants in that lane SKIPPED; the other lane continues.
- Whole-job failure only from source download failure (nothing else to do).

## 7. Quantization catalog (v1)

**GGUF K-quants** (`llama-quantize`, no imatrix):
Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K

**GGUF I-quants** (`llama-quantize --imatrix`, requires imatrix pre-step):
IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ3_XXS, IQ3_XS, IQ3_S, IQ3_M, IQ4_XS, IQ4_NL

**GGUF misc**:
Q8_0 (K-quant style but often used as a reference), F16, BF16 (repacks; no calibration).

**GGUF multimodal**:
mmproj-f16 (auto-included when the source model is Mistral3ForConditionalGeneration).

**safetensors** (llm-compressor):
- W4A16 GPTQ, symmetric, group_size=128 (the operator's current backend deliverable recipe).
- W4A16 AWQ, asymmetric, group_size=128 (fallback / community-preferred).
- NVFP4 (4-bit weights + 4-bit activations, Blackwell only).
- FP8 E4M3 (weight-only 8-bit float; Hopper+ compute, works on Ampere as weight storage).
- FP8 E5M2 (similarly).

Total v1 catalog: **9 GGUF K-quants + 12 GGUF I-quants + 3 GGUF misc + optional mmproj + 5 safetensors = 30 selectable outputs.** The UI presents them grouped.

## 8. Calibration data model

Three sources, operator picks one per job:

1. **Bundled generic corpus** — pileval subset + wikitext samples baked into the image (~1000 samples, ~2 MB). Zero friction; suboptimal for domain-specific models.
2. **Upload JSONL** — operator uploads a calibration file (OpenAI-messages JSONL format matching the operator's `frndobrain_quantize.py` expectations). Used as-is.
3. **HF dataset ID** — operator enters a repo id (e.g., `bigblueceiling/frndobrain-cal-v1`). Downloaded using the same HF token supplied for the job. Falls back to option 1 if the dataset is unreachable.

The chosen calibration source feeds both Lane A (llm-compressor) and Lane B's imatrix step, uniformly, so both lanes see the same distribution.

## 9. HuggingFace integration

### Token lifecycle
- Held in the FastAPI process's memory only. Not written to disk, not exported as an env var visible to subprocesses (passed via `huggingface_hub` API calls directly).
- Wiped from memory when a job finishes (success or failure).
- Next job requires re-entry.

### Uploads
- Each safetensors quant → own repo: `<owner>/<sourcemodel-basename>-<FORMAT>` (e.g. `-GPTQ`, `-AWQ`, `-NVFP4`, `-FP8-E4M3`, `-FP8-E5M2`).
- All GGUFs (K + I + misc + mmproj) → single repo: `<owner>/<sourcemodel-basename>-GGUF`. Each quant is a separate file within that repo.
- Repo names editable per-row in the UI before Launch. Defaults are auto-generated as above.
- Public/private toggle at repo creation time. Applies to all repos created by the job.
- Uploads use `huggingface_hub.HfApi.upload_folder` (safetensors, mmproj+lm gguf combined per dir) or `.upload_file` (single-file GGUFs).

### README update
- After each safetensors quant is fully uploaded (or after the GGUF repo is populated), attempt a direct commit to the source repo's `main` branch appending or replacing a `## Quantizations` section.
- If the token cannot write to the source repo (403), log a warning in the job report and skip. Do not attempt PRs in v1.
- Idempotent: on subsequent runs, the whole `## Quantizations` section is replaced. Section is a canonical Markdown table (Format | Repo | Notes).

## 10. UI surface

### Setup page (single form, one URL: `GET /`)
- **Source model:** HF repo id, e.g. `bigblueceiling/frndobrain-empathetic-r5-Instruct`. Free text; validated on Launch by attempting a HEAD against `hf://` metadata.
- **HF token:** password field. Cleared on job end.
- **Calibration source dropdown:** [Bundled generic] | [Upload JSONL] | [HF dataset ID]. Conditional file input / text input appear based on selection.
- **Quant selection:** grouped checkbox tree —
  - GGUF K-quants (9 items)
  - GGUF I-quants (12 items, header shows "needs imatrix pre-step")
  - GGUF misc (Q8_0, F16, BF16 — 3 items)
  - GGUF mmproj (1 item, only shown if source is detected multimodal — auto-check by default)
  - safetensors (5 items, NVFP4 grayed with a Blackwell-required note if `nvidia-smi` reports SM<12)
- **Upload owner:** HF org/username. Defaults to the token owner's namespace after token entry (fetched lazily on token change via `whoami`).
- **Repo name previews:** editable table showing each output's target repo name, auto-generated. Operator can override any row.
- **Public/private:** radio.
- **Update source README:** checkbox, default ON. Grays out with a warning if the operator's namespace doesn't own the source repo (heuristic: `owner-of-source != token-owner`).
- **[Launch]** button — POST `/jobs`.

### Job view (per running job, one URL: `GET /jobs/<job_id>`)
- Overall progress bar + estimated total time (based on median time per quant on similar-size models — rough).
- Per-quant table with columns: name, lane (A/B), status, size (once produced), elapsed, [expand].
- Expanded row shows a live tail of that quant's stdout/stderr (~last 50 lines), streamed via SSE.
- Two "current activity" strips showing what Lane A and Lane B are doing right now.
- Final job report block at the bottom once done: successes, failures with reasons, total elapsed, README-update outcome per repo.

### History (one URL: `GET /jobs`)
- Table of jobs run in this pod session (in-memory list, wiped on restart). Columns: job id, source model, launched at, status, links to job view.

## 11. Failure modes and handling

| Failure | Blast radius | Handling |
|---|---|---|
| Source model 404 or 401 | Whole job | Abort before job start; UI shows error. |
| Source download OOM (disk) | Whole job | Abort with clear disk-size message. |
| Lane A: GPU load OOM | Lane A only | Mark all safetensors quants SKIPPED with reason; Lane B continues. |
| Lane B: BF16 GGUF conversion fails | Lane B only | Mark all GGUFs SKIPPED; Lane A continues. |
| Individual quant fails (e.g., NVFP4 requires Blackwell, AWQ pattern-match error) | That quant | Mark FAILED with reason; continue with other quants in same lane. |
| Upload fails (network, rate limit) | That upload | Retry with exponential backoff up to 3 attempts; if all fail, mark upload-FAILED; keep local file for manual retry. |
| README commit fails (403, conflict) | README only | Log warning in job report; quants stay uploaded. |
| Pod restarts mid-job | Whole job | Job lost; artifacts already uploaded stay uploaded, in-progress quants are gone. Operator restarts as a new job. |

## 12. Security considerations

- Token in-memory only, wiped on job end. Never logged.
- Web UI has no auth; the pod's port exposure is the trust boundary (RunPod SSH tunnel or their access-URL proxy).
- Uploaded quants inherit the operator's chosen public/private setting; no ambient defaults.
- The Docker image contains no operator credentials; every token is entered per job.
- Calibration JSONL uploads are held on the pod's ephemeral disk during the job and deleted at job end.

## 13. Observability

- Every job's per-quant logs held in-memory during the job (last N lines per quant, ~50 KB per quant). Full logs saved to `/workspace/logs/<job_id>/<quant>.log` on the pod so they can be `docker cp`'d out if desired.
- No external log shipping. No metrics endpoint. Pod-local only.

## 14. Explicitly deferred to v2 or later

- Non-Mistral architectures.
- EXL2 quants.
- Batch job submission (multiple source models in one launch).
- Auto-detection of "optimal" quant recipes for a given target GPU.
- Per-user auth / multi-tenancy.
- Retrying failed quants without a full new job.
- Automatic pod provisioning.
- Job persistence across pod restarts.
- Progress ETA better than "linear extrapolation from median quant time."
